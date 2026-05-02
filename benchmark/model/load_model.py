import logging


def load_model(train_dataset, device='cpu', **kwargs):
    model_name = kwargs['model_name']

    input, target = train_dataset[0] # T, S, S, C
    input_shape = input.shape
    output_shape = target.shape

    logging.debug(f"Loading model {model_name} with input shape {input_shape} and output shape {output_shape}")
    if model_name == 'fno':
        from benchmark.model.fno import FNO3d
        model = FNO3d(
            modes1=kwargs['modes1'], 
            modes2=kwargs['modes2'], 
            modes3=kwargs['modes3'], 
            n_layers=kwargs['n_layers'],
            width=kwargs['width'], 
            shape_in=input_shape, 
            shape_out=output_shape
            ).to(device)

    elif model_name == 'cno':
        from benchmark.model.cno import CNO3d
        if output_shape[0] > input_shape[0] and output_shape[0] % input_shape[0] == 0:
            out_dim_mult = output_shape[0] // input_shape[0]
        elif output_shape[0] == input_shape[0]:
            out_dim_mult = 1
        else:
            raise ValueError(f"Output shape {output_shape[1]} is not a multiple of input shape {input_shape[1]}")
        
        model = CNO3d(
                        in_dim=input_shape[-1],
                        out_dim=output_shape[-1],
                        out_dim_mult=out_dim_mult,
                        in_size=input_shape[2],
                        N_layers=kwargs['N_layers'],
                    ).to(device)

    elif model_name == 'dpot':
        from benchmark.model.dpot import DPOT
        model = DPOT(
            shape_in=input_shape,
            shape_out=output_shape,
            img_size=kwargs["img_size"],
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            in_timesteps=kwargs["in_timesteps"],
            out_timesteps=kwargs["out_timesteps"],
            patch_size=kwargs["patch_size"],
            embed_dim=kwargs["embed_dim"],
            depth=kwargs["depth"],
            n_blocks=kwargs["n_blocks"],
            modes=kwargs["modes"],
            mlp_ratio=kwargs["mlp_ratio"],
            out_layer_dim=kwargs["out_layer_dim"],
            normalize=kwargs["normalize"],
            act=kwargs["act"],
            time_agg=kwargs["time_agg"],
            n_cls=kwargs["n_cls"],
            model_type=kwargs["model_type"],
            checkpoint_path=kwargs["checkpoint_path"],
        ).to(device)
    elif model_name == 'deeponet':
        from benchmark.model.deeponet import DeepONet
        
        model = DeepONet(
            shape_in=input_shape,
            shape_out=output_shape,
            input_channels=input_shape[-1],
            output_channels=output_shape[-1],
            p=kwargs['p'],
            dropout_rate=kwargs['dropout_rate'],
            device=device
        ).to(device)

    elif model_name == 'transolver':
        from benchmark.model.transolver_libs.Transolver_Structured_Mesh_3D import Model
        
        model = Model(
            space_dim=kwargs['space_dim'], n_layers=kwargs['n_layers'], n_hidden=kwargs['n_hidden'], n_head=kwargs['n_head'],
            H=kwargs['H'], W=kwargs['W'], D=kwargs['D'], Time_Input=False, unified_pos=False,
            fun_dim=kwargs['fun_dim'], out_dim=kwargs['out_dim'], ref=kwargs['ref'],dropout=kwargs['dropout'], act=kwargs['act'], mlp_ratio=kwargs['mlp_ratio'], slice_num=kwargs['slice_num']
        ).to(device)
    
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model