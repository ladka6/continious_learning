def get_model(model_name, args):
    name = model_name.lower()
    if name == 'tosca':
        from models.tosca import Learner
    elif name == 'adaptive_gating_tosca':
        from models.adaptive_gating_tosca import Learner
    else:
        raise NotImplementedError("Unknown model: {}".format(model_name))
    return Learner(args)
