import sugartensor as tf


__author__ = 'namju.kim@kakaobrain.com'


cat_dim = 10   # total categorical factor
con_dim = 2    # total continuous factor
rand_dim = 18  # total random latent dimension


#
# create generator & discriminator function
#

def generator(tensor):

    # reuse flag
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0

    with tf.sg_context(name='generator', size=(4, 1), stride=(2, 1), act='leaky_relu', bn=True, reuse=reuse):
        res = (tensor
               .sg_dense(dim=1024, name='fc1')
               .sg_dense(dim=48 * 1 * 128, name='fc2')
               .sg_reshape(shape=(-1, 48, 1, 128))
               .sg_upconv(dim=64, name='conv1')
               .sg_upconv(dim=32, name='conv2')
               .sg_upconv(dim=2, act='sigmoid', bn=False, name='conv3'))
    return res


def discriminator(tensor):

    # reuse flag
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0

    with tf.sg_context(name='discriminator', size=(4, 1), stride=(2, 1), act='leaky_relu', bn=False, reuse=reuse):
        # shared part
        shared = (tensor
                  .sg_conv(dim=32, name='conv1')
                  .sg_conv(dim=64, name='conv2')
                  .sg_conv(dim=128, name='conv3')
                  .sg_flatten()
                  .sg_dense(dim=1024, name='fc1'))

        # discriminator end
        disc = shared.sg_dense(dim=1, act='linear', bn=False, name='disc').sg_squeeze()

        # shared recognizer part
        recog_shared = shared.sg_dense(dim=128, name='recog')

        # categorical auxiliary classifier end
        cat = recog_shared.sg_dense(dim=cat_dim, act='linear', bn=False, name='cat')

        # continuous auxiliary classifier end
        con = recog_shared.sg_dense(dim=con_dim, act='sigmoid', bn=False, name='con')

        return disc, cat, con