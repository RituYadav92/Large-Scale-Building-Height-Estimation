from .tswin_unet import TSwinUnet


def define_model(configs):
    
    if configs.name == 'tswin_unet':
        model = TSwinUnet(**configs.params)
    else:
        raise NotImplementedError(f'unknown model name {configs.name}')

    return model
