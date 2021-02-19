# -*- coding: utf-8 -*-

"""
Simple Image Captioning Implementation with GPT-2 and MobileNet V3

"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import mymodels as mm


class Attention(mm.Attention):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def propagating(self, x, past=None, image=None, training=False):
        q1, k1, v1 = tf.split(self.wq(self.norm(x)), 3, 2)
        k1 = tf.concat([image[0], k1], -2) if image is not None else k1
        v1 = tf.concat([image[1], v1], -2) if image is not None else v1
        k2 = tf.concat([past[:, 0], k1], -2) if past is not None else k1
        v2 = tf.concat([past[:, 1], v1], -2) if past is not None else v1
        m1 = tf.range(tf.shape(q1)[1])[:, tf.newaxis] < tf.range(tf.shape(k2)[1])-tf.shape(k2)[1]+tf.shape(q1)[1]
        x1, a1 = self.calculating(q1, k2, v2, m1, True, training)
        x1 = tf.reshape(tf.transpose(x1, [0, 2, 1, 3]), [-1, tf.shape(x1)[2], self.dim])
        return x+self.drop(self.dense(x1), training=training), a1, tf.stack([k1, v1], 1)


class TransEncoder(mm.TransEncoder):
    def __init__(self, **kwargs):
        super(TransEncoder, self).__init__(**kwargs)
        self.att = Attention(bname=kwargs['bname'], lname=kwargs['lname'], head=kwargs['head'], size=kwargs['size'],
                             attdrop=0., drop=0., eps=1e-6, mlm=False)

    def propagating(self, x, past=None, image=None, training=False):
        x1, a1, p1 = self.att.propagating(x, past, image, training)
        x2 = self.drop(self.dense2(self.dense1(x1 if self.mlm else self.norm(x1))), training=training)
        return self.norm(x1+x2) if self.mlm else (x1+x2, p1)


class GPT2(mm.GPT):
    def __init__(self, **kwargs):
        super(GPT2, self).__init__(**kwargs)
        self.encoder = [TransEncoder(
            bname='model/h'+str(i1),
            lname=['/attn/c_attn', '', '', '/attn/c_proj', '/lnorm_1', '/mlp/c_fc', '/mlp/c_proj', '/lnorm_2'],
            head=self.param['n_head'],
            size=self.param['n_embd']//self.param['n_head'],
            dff=self.param['n_embd']*4,
            act=self.param.get('activation', mm.gelu_activating),
            mlm=False) for i1 in range(self.param['n_layer'])]

    def propagating(self, x, pos=None, past=None, image=None, training=False, softmax=True):
        p1, p2 = (tf.unstack(past, axis=1) if past is not None else [None]*self.param['n_layer']), []
        t1 = pos if pos is not None else tf.repeat([past.shape[3]], x.shape[0], 0) if past is not None else None
        x1 = self.embedding.propagating(x, None, t1, training)

        for i1 in range(self.param['n_layer']):
            x1, a1 = self.encoder[i1].propagating(x1, p1[i1], None if image is None else image[i1], training)
            p2.append(a1)

        x1, h1 = self.norm(x1), tf.stack(p2, 1)
        x2 = tf.matmul(x1, self.embedding.emb, transpose_b=True)
        return tf.nn.softmax(x2) if softmax else x2, x1, h1

    @tf.function
    def calling(self, x, image):
        return self.propagating(x, None, None, image, False, False)

    @tf.function(experimental_relax_shapes=True)
    def iterating(self, x, pos, past):
        return self.propagating(x, pos, past, None, False, False)

    @tf.function(experimental_relax_shapes=True)
    def calculating(self, score, cur, beam, k, p, length, penalty, first):
        return super(GPT2, self).calculating(score, cur, beam, k, p, length, penalty, first)

    def generating(self, x, pos, image=None, beam=5, k=1, p=0.9, temp=1.0, penalty=1.0, maxlen=10, best=False):
        x1, b1, p1 = x, x.shape[0], tf.repeat(pos, beam, 0)
        scor1, leng1, past1, fini1, pred1 = 0., tf.cast(pos, tf.float32)+1., None, None, None
        list1, list2, i1 = np.arange(b1), [None]*b1, 0
        mask1 = tf.repeat(tf.one_hot([self.eos], self.param['n_vocab'], 0., self.ninf), b1*beam, 0)
        appe1 = tf.one_hot([self.eos], self.param['n_vocab'], 0., 1.)

        while i1 < maxlen and len(list1) > 0:
            x2, _, h1 = self.iterating(x1, p1+i1, past1) if i1 else self.calling(x1, image)
            s1 = tf.nn.log_softmax(tf.squeeze(x2/temp, 1)+fini1*mask1 if i1 else tf.gather_nd(x2/temp, pos, 1))
            x1, h2, scor1, leng1 = self.calculating(scor1+s1, s1, beam, k, p, leng1+appe1, penalty, i1 == 0)
            past1 = tf.gather(tf.concat([past1, h1], -2) if i1 else h1, h2)
            pred1, i1 = tf.concat([tf.gather(pred1, h2), x1], -1) if i1 else x1, i1+1
            e1, e2, fini1, list1, list2 = self.updating(x1, pred1, list1, list2, beam, i1 == maxlen)

            if len(e1) > 0 and i1 < maxlen:
                x1, p1, past1, scor1, leng1, mask1, fini1, pred1 = [
                    tf.gather(j1, e2) for j1 in [x1, p1, past1, scor1, leng1, mask1, fini1, pred1]]

        return [[i1[0]] for i1 in list2] if best else list2


class MobileNet3(mm.MobileNet):
    def __init__(self, **kwargs):
        super(MobileNet3, self).__init__(**kwargs)

    @tf.function(experimental_relax_shapes=True)
    def propagating(self, x, training=False):
        x1 = self.act(self.norm1(self.conv1(x), training=training))

        for i1 in self.encoder:
            x1 = i1.propagating(x1, training)

        return self.act(self.norm2(self.conv2(x1), training=training))


class ConvBlock(keras.layers.Layer):
    def __init__(self, ln, channel, act, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.act = {'hswish': mm.hswish_activating, 'relu': tf.nn.relu}[act]
        self.conv1 = keras.layers.Conv2D(channel, 1, 1, 'same', use_bias=False, name=ln+'conv/')
        self.conv2 = keras.layers.DepthwiseConv2D(3, 1, 'same', use_bias=False, name=ln+'depthwise/')
        self.norm1 = mm.Normalization('instance', init=tf.random_normal_initializer(1., 0.02), name=ln+'cnorm/')
        self.norm2 = mm.Normalization('instance', init=tf.random_normal_initializer(1., 0.02), name=ln+'dnorm/')

    def propagating(self, x):
        x1 = self.act(self.norm1(self.conv1(x)))
        return x+self.act(self.norm2(self.conv2(x1)))


class Caption(keras.Model):
    def __init__(self, vocab, gconfig, gckpt, mckpt=None, cin=576, **kwargs):
        super(Caption, self).__init__(**kwargs)
        self.wk, self.wv, self.norm, self.enc = [], [], [], []
        self.tokenizer = mm.Tokenizer(False, False, False)
        self.tokenizer.loading(vocab)
        self.gpt = GPT2(config=gconfig)
        self.gpt.loading(gckpt)

        for i1 in range(self.gpt.param['n_layer']):
            self.wk.append(keras.layers.Dense(self.gpt.param['n_embd'], name='ienc/k'+str(i1)))
            self.wv.append(keras.layers.Dense(self.gpt.param['n_embd'], name='ienc/v'+str(i1)))
            self.norm.append(keras.layers.LayerNormalization(-1, 1e-6, name='ienc/nm'+str(i1)))
            self.enc += [ConvBlock('ienc/l'+str(i1), cin, 'hswish')]

        if mckpt:
            self.mobile = MobileNet3(alpha=1, caxis=-1, cate=1001)
            self.mobile.loading(mckpt)

    def encoding(self, feature):
        x1, s1, h1 = feature, tf.shape(feature), []

        for i1 in range(self.gpt.param['n_layer']):
            x1 = self.enc[i1].propagating(x1)
            x2 = self.norm[i1](tf.reshape(x1, [-1, s1[1]*s1[1], s1[-1]]))
            h1.append([self.wk[i1](x2), self.wv[i1](x2)])

        return h1

    def generating(self, image, maxlen, minlen):
        x1, s1 = self.encoding(self.mobile.propagating(image)), int(tf.shape(image)[0])
        x2, p1, d1 = np.array([[self.gpt.eos]]*s1), np.array([[0]]*s1), ['']*s1
        g1 = self.gpt.generating(x2, p1, x1, maxlen=maxlen, best=False)

        for i1, j1 in enumerate(g1):
            clen1 = 0

            for k1 in j1:
                leng1, text1 = len(k1[:(k1+[self.gpt.eos]).index(self.gpt.eos)]), self.tokenizer.decoding(k1)
                d1[i1], clen1 = (text1, leng1) if leng1 > min(minlen-1, clen1) else (d1[i1], clen1)

                if leng1 >= minlen:
                    break

        return d1, g1

    def call(self, x, text=None, training=False, softmax=True, **kwargs):
        return self.gpt.propagating(text, image=self.encoding(x), training=training, softmax=softmax)

    def get_config(self):
        return self.config
