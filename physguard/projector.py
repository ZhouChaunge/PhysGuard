"""
Null Space Projector for Sim-to-Real Fine-tuning.

Core idea: compute the "important parameter subspace" from simulated pre-training,
then project fine-tuning gradients onto its orthogonal complement (null space),
so that adaptation to real data does not destroy learned physics.

Supports:
  - Fisher Information Matrix (FIM) based subspace estimation
  - Layer-wise adaptive null space projection
  - Progressive relaxation of protection strength
  - Spectral (frequency-domain) null space for FNO models
"""

import gc
import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm


class NullSpaceProjector:
    """
    Computes and stores the null-space projection matrices per parameter group.

    Usage:
        1. After pre-training, call `compute_projection(model, dataloader, ...)` to
           build the projection matrices from simulated data.
        2. During fine-tuning, call `project_gradients(model)` after loss.backward()
           and before optimizer.step() to project gradients into the null space.
    """

    def __init__(
        self,
        n_components: int = 200,
        alpha: float = 1.0,
        progressive: bool = False,
        total_steps: int = 1000,
        beta_min: float = 0.3,
        layer_wise: bool = True,
        device: str = "cuda",
        protected_layers: Optional[List[str]] = None,
        variance_threshold: Optional[float] = 0.9,
    ):
        """
        Args:
            n_components: Maximum number of top eigenvectors to consider. When
                variance_threshold is set, this acts as an upper-bound cap;
                the actual k per layer is determined adaptively.
            alpha: Global protection strength. 1.0 = full null-space projection.
            progressive: Whether to progressively relax the constraint over training.
            total_steps: Total fine-tuning steps (for progressive schedule).
            beta_min: Minimum protection strength at the end of progressive relaxation.
            layer_wise: Whether to compute per-layer projections (True) or a single global one.
            device: Device for computation.
            protected_layers: Optional list of parameter name substrings to protect.
                Only parameters whose name contains at least one of these substrings
                will have null-space projection applied.
                e.g. ["spectral_convs"] protects only the Fourier weight tensors in FNO,
                leaving BN / fc layers free to adapt.
                If None or empty, ALL parameters are protected (original behaviour).
            variance_threshold: If set (e.g. 0.9), adaptively choose k per layer as
                the minimum number of eigenvectors that explain at least this fraction
                of the total Fisher variance (cumulative eigenvalue ratio >= threshold).
                n_components is used as an upper-bound cap.
                If None, always use exactly n_components eigenvectors (legacy behaviour).
        """
        self.n_components = n_components
        self.alpha = alpha
        self.progressive = progressive
        self.total_steps = total_steps
        self.beta_min = beta_min
        self.layer_wise = layer_wise
        self.device = device
        self.variance_threshold = variance_threshold
        # Normalise: None / empty list both mean "protect everything"
        self.protected_layers: Optional[List[str]] = protected_layers if protected_layers else None

        # Stores per-parameter projection information
        # key: parameter name, value: dict with 'U' (top eigenvectors) and optional metadata
        self.projections: Dict[str, Dict] = {}
        self._step = 0

    def compute_projection(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        data_normalizer=None,
        max_samples: int = 500,
        loss_fn=None,
    ):
        """
        Compute the Fisher Information Matrix (empirical) for each parameter group,
        then extract the top-k eigenvectors to define the "important subspace".

        The null-space projection matrix for parameter p is:
            P_null = I - U @ U^T
        where U contains the top-k eigenvectors of the FIM for that parameter.

        We store U per parameter to save memory (no need to store the full projection matrix).

        Args:
            model: The pre-trained model.
            dataloader: DataLoader over simulated (numerical) data.
            data_normalizer: Optional data normalizer (preprocess input/target).
            max_samples: Maximum number of samples to use for FIM estimation.
            loss_fn: Loss function. If None, uses model.train_loss.
        """
        model.eval()

        # Collect parameter names and shapes for parameters that require grad
        param_names = []
        param_shapes = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_names.append(name)
                param_shapes[name] = param.shape

        logging.info(f"Computing null-space projections for {len(param_names)} parameters...")

        if self.layer_wise:
            self._compute_layerwise(model, dataloader, data_normalizer, max_samples, loss_fn, param_names)
        else:
            self._compute_global(model, dataloader, data_normalizer, max_samples, loss_fn, param_names)

        logging.info(f"Null-space projections computed. Subspace dim = {self.n_components}")

    def _compute_layerwise(self, model, dataloader, data_normalizer, max_samples, loss_fn, param_names):
        """
        Compute per-layer Fisher top eigenvectors via the Dual SVD / Gram trick.

        Instead of building the d×d Fisher matrix F = (1/N) G^T G (impossible
        for large d), we build the N×N Gram matrix K = G G^T and use:
            eigh(K) -> eigvecs u_k   (N×k)
            V = G^T @ u_k            (d×k, Fisher eigenvectors in param space)
        
        This exploits the fact that G^T G and G G^T share non-zero eigenvalues.
        Memory: O(N·d) for storing gradients on CPU, O(N²) for Gram matrix.
        """
        # Phase 1: collect per-sample gradient vectors (stored on CPU to save GPU mem)
        grad_accum: Dict[str, List[torch.Tensor]] = {name: [] for name in param_names}
        n_samples = 0

        for batch_data in tqdm(dataloader, desc="Collecting gradients"):
            if n_samples >= max_samples:
                break

            input_data, target_data = batch_data
            if data_normalizer is not None:
                input_data, target_data = data_normalizer.preprocess(input_data, target_data)
            else:
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

            b = input_data.size(0)

            for i in range(b):
                if n_samples >= max_samples:
                    break

                model.zero_grad()
                inp_i = input_data[i:i+1]
                tgt_i = target_data[i:i+1]

                if loss_fn is not None:
                    pred = model(inp_i)
                    loss = loss_fn(pred, tgt_i).mean()
                else:
                    loss = model.train_loss(inp_i, tgt_i).mean()

                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Store flattened gradient on CPU in float16 to halve memory
                        g = param.grad.detach().cpu().reshape(-1)
                        if g.is_complex():
                            # Concatenate real and imaginary parts into a real vector
                            g = torch.cat([g.real, g.imag], dim=0)
                        grad_accum[name].append(g.half())

                n_samples += 1

        logging.info(f"Collected gradients from {n_samples} samples.")

        # Estimate total gradient storage for memory reporting
        total_grad_bytes = sum(
            sum(g.numel() * g.element_size() for g in glist)
            for glist in grad_accum.values() if glist
        )
        logging.info(f"Gradient storage: {total_grad_bytes / 1e9:.1f} GB CPU RAM")

        # Phase 2: for each parameter, compute top-k Fisher eigenvectors via Gram trick
        # Sort by gradient matrix size (largest first) to free most memory early
        sorted_names = sorted(
            param_names,
            key=lambda n: grad_accum[n][0].numel() if grad_accum.get(n) else 0,
            reverse=True,
        )

        for name in tqdm(sorted_names, desc="Computing null-space projections (dual SVD)"):
            grads = grad_accum.pop(name, None)  # pop to remove reference immediately
            if not grads:
                continue

            # G: [N, d] gradient matrix on CPU (already real after complex split)
            G = torch.stack(grads, dim=0).float()  # [N, d]
            del grads  # free individual tensor list before SVD
            gc.collect()

            param_dim = G.shape[1]
            k_max = min(self.n_components, param_dim, n_samples)

            if k_max == 0:
                del G
                continue

            if param_dim <= 2 * k_max:
                # Very small parameter — protect all dimensions
                k = k_max
                U = torch.eye(param_dim)[:, :k]
            else:
                # Dual trick: K = G @ G^T is only [N, N] (e.g. 200×200)
                K = G @ G.T  # [N, N]

                # Eigen-decompose the small Gram matrix (ascending order)
                eigenvalues, eigvecs = torch.linalg.eigh(K)

                # --- Adaptive k selection via cumulative variance threshold ---
                if self.variance_threshold is not None:
                    # eigenvalues are ascending; reverse for descending
                    evals_desc = eigenvalues.flip(0).clamp(min=0.0)  # [N]
                    total_var = evals_desc.sum()
                    if total_var > 0:
                        cum_ratio = evals_desc.cumsum(0) / total_var
                        # k = smallest index where cumulative ratio >= threshold
                        above = (cum_ratio >= self.variance_threshold).nonzero(as_tuple=False)
                        k_adaptive = int(above[0].item()) + 1 if len(above) > 0 else len(evals_desc)
                    else:
                        k_adaptive = k_max
                    k = min(k_adaptive, k_max)  # cap at n_components
                else:
                    k = k_max  # legacy: fixed k

                # Take top-k (largest eigenvalues = last k columns)
                top_eigvecs = eigvecs[:, -k:]  # [N, k]
                del eigenvalues, eigvecs, K

                # Map back to parameter space: V = G^T @ u, then normalize
                U = G.T @ top_eigvecs  # [d, k]
                del top_eigvecs
                norms = U.norm(dim=0, keepdim=True).clamp(min=1e-8)
                U = U / norms  # orthonormal columns

            del G
            gc.collect()

            var_pct = k / param_dim * 100
            logging.debug(
                f"  {name}: param_dim={param_dim}, k={k} "
                f"({var_pct:.3f}% of param space protected)"
            )

            # Store on CPU to save GPU memory (moved to GPU on-the-fly during projection)
            self.projections[name] = {
                'U': U.cpu(),  # [d, k], stored on CPU
                'dim': param_dim,
            }

    def _compute_global(self, model, dataloader, data_normalizer, max_samples, loss_fn, param_names):
        """Compute a single global Fisher and extract top eigenvectors (for smaller models)."""
        # Concatenate all params into a single vector for global FIM
        total_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if total_dim > 50000:
            logging.warning(
                f"Global mode with {total_dim} params is memory-intensive. "
                f"Consider using layer_wise=True or reducing n_components."
            )

        fisher = torch.zeros(total_dim, total_dim, device=self.device)
        n_samples = 0

        for batch_data in tqdm(dataloader, desc="Computing Fisher (global)"):
            if n_samples >= max_samples:
                break

            input_data, target_data = batch_data
            if data_normalizer is not None:
                input_data, target_data = data_normalizer.preprocess(input_data, target_data)
            else:
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

            b = input_data.size(0)
            for i in range(b):
                if n_samples >= max_samples:
                    break

                model.zero_grad()
                inp_i = input_data[i:i+1]
                tgt_i = target_data[i:i+1]

                if loss_fn is not None:
                    pred = model(inp_i)
                    loss = loss_fn(pred, tgt_i).mean()
                else:
                    loss = model.train_loss(inp_i, tgt_i).mean()

                loss.backward()

                # Gather all gradients into a single vector
                grads = []
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grads.append(param.grad.detach().reshape(-1))
                g = torch.cat(grads)
                fisher += g.unsqueeze(1) * g.unsqueeze(0)
                n_samples += 1

        fisher /= n_samples
        logging.info(f"Global Fisher computed from {n_samples} samples.")

        k = min(self.n_components, total_dim)
        fisher += 1e-8 * torch.eye(total_dim, device=self.device)
        U, S, _ = torch.svd_lowrank(fisher, q=k, niter=5)

        # Distribute back to per-parameter projections
        offset = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                dim = param.numel()
                U_local = U[offset:offset+dim, :]  # [dim, k]
                self.projections[name] = {
                    'U': U_local.cpu(),
                    'dim': dim,
                }
                offset += dim

    def get_protection_strength(self) -> float:
        """Get current protection strength, accounting for progressive relaxation."""
        if not self.progressive:
            return self.alpha

        # Cosine annealing from alpha to beta_min
        progress = min(self._step / max(self.total_steps, 1), 1.0)
        beta = self.beta_min + 0.5 * (self.alpha - self.beta_min) * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        return beta.item()

    def project_gradients(self, model: nn.Module):
        """
        Project the gradients of all parameters into the null space of the
        pre-trained important subspace.

        Call this AFTER loss.backward() and BEFORE optimizer.step().
        
        For each parameter with gradient g:
            g_projected = g - alpha * U @ (U^T @ g)
        
        This removes the component of g that lies in the important subspace,
        keeping only the null-space component.
        """
        if not self.projections:
            return  # No projections computed yet

        beta = self.get_protection_strength()

        for name, param in model.named_parameters():
            # Skip parameters not in the protected-layer list (if one is specified)
            if self.protected_layers is not None:
                if not any(pat in name for pat in self.protected_layers):
                    continue

            if param.requires_grad and param.grad is not None and name in self.projections:
                proj_info = self.projections[name]
                # Move U to the same device as grad on-the-fly, then discard
                U = proj_info['U'].to(param.grad.device)  # [dim, k]

                g = param.grad.detach().reshape(-1)  # flatten gradient

                # Handle complex parameters (e.g. SpectralConv3d weights)
                is_complex = g.is_complex()
                if is_complex:
                    g = torch.cat([g.real, g.imag], dim=0)  # [2*numel]

                # Project out the important subspace component:
                # g_null = g - beta * U @ (U^T @ g)
                g_proj = U.T @ g           # [k]
                g_important = U @ g_proj   # [dim]
                g_null = g - beta * g_important

                if is_complex:
                    # Reconstruct complex gradient
                    half = g_null.shape[0] // 2
                    g_null = torch.complex(g_null[:half], g_null[half:])

                param.grad.data = g_null.reshape(param.shape)
                del U  # free GPU copy immediately

        self._step += 1

    def save(self, path: str):
        """Save the projections to disk."""
        save_dict = {
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold,
            'alpha': self.alpha,
            'progressive': self.progressive,
            'total_steps': self.total_steps,
            'beta_min': self.beta_min,
            'layer_wise': self.layer_wise,
            'protected_layers': self.protected_layers,
            'projections': {
                name: {
                    'U': info['U'].cpu(),
                    'dim': info['dim'],
                }
                for name, info in self.projections.items()
            },
        }
        torch.save(save_dict, path)
        logging.info(f"Null-space projections saved to {path}")

    def load(self, path: str):
        """Load projections from disk.
        
        NOTE: Only projection matrices and n_components are restored from file.
        Runtime hyperparams (alpha, protected_layers, progressive, etc.) keep
        their values as set in __init__ / via CLI, so ablation experiments that
        share the same projection file can each use different settings.
        """
        save_dict = torch.load(path, map_location='cpu')
        self.n_components = save_dict['n_components']
        # Do NOT restore alpha / protected_layers / progressive / variance_threshold
        # from file — those are runtime params that must come from the current
        # experiment's config.  Restoring them would silently override ablation settings.
        self.projections = {
            name: {
                'U': info['U'].cpu(),
                'dim': info['dim'],
            }
            for name, info in save_dict['projections'].items()
        }
        logging.info(f"Null-space projections loaded from {path} "
                     f"(n_components={self.n_components}, alpha={self.alpha}, "
                     f"protected_layers={self.protected_layers})")
