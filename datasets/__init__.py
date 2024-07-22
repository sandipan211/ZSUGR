from .caddy import build as build_caddy

def build_dataset(image_set, args):
    if args.dataset == 'CADDY':
        return build_caddy(image_set, args)

    raise ValueError(f'dataset {args.dataset} not supported')

