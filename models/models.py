
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'multi_half' or opt.model == 'multi_half_style_c':
        from .multi_half_gan import MultiHalfGanModel
        model = MultiHalfGanModel()
    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
