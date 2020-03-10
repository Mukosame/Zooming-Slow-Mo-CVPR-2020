import models.modules.Sakuya_arch as Sakuya_arch

####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'LunaTokis':
        netG = Sakuya_arch.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'])                              
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
