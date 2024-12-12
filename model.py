from __future__ import print_function

import os
import time
import random

from PIL import Image
import tensorflow as tf
import numpy as np

from utils import *


def concat(layers):
    return tf.concat(layers, axis=3)


class DecomNet(tf.keras.layers.Layer):
    def __init__(self, layer_num, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num
        self.channel = channel
        self.kernel_size = kernel_size

        # 初始化网络层
        self.shallow_feature_extraction = tf.keras.layers.Conv2D(
            channel, kernel_size * 3, padding='same', activation=None, name="DecomNet_shallow_feature_extraction"
        )
        self.activated_layers = [
            tf.keras.layers.Conv2D(
                channel, kernel_size, padding='same', activation='relu', name=f"DecomNet_activated_layer_{i}"
            ) for i in range(layer_num)
        ]
        self.recon_layer = tf.keras.layers.Conv2D(
            4, kernel_size, padding='same', activation=None, name="DecomNet_recon_layer"
        )

    def call(self, input_im):
        # 处理输入
        input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
        input_im = concat([input_max, input_im])

        # 前向传播
        x = self.shallow_feature_extraction(input_im)
        for layer in self.activated_layers:
            x = layer(x)
        x = self.recon_layer(x)

        # 分离输出
        R = tf.sigmoid(x[:, :, :, 0:3])
        L = tf.sigmoid(x[:, :, :, 3:4])
        return R, L


class RelightNet(tf.keras.layers.Layer):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size

        # 初始化网络层
        self.conv0 = tf.keras.layers.Conv2D(
            channel, kernel_size, padding='same', activation=None, name="RelightNet_conv0"
        )
        self.conv1 = tf.keras.layers.Conv2D(
            channel, kernel_size, strides=2, padding='same', activation='relu', name="RelightNet_conv1"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            channel, kernel_size, strides=2, padding='same', activation='relu', name="RelightNet_conv2"
        )
        self.conv3 = tf.keras.layers.Conv2D(
            channel, kernel_size, strides=2, padding='same', activation='relu', name="RelightNet_conv3"
        )
        self.deconv1 = tf.keras.layers.Conv2D(
            channel, kernel_size, padding='same', activation='relu', name="RelightNet_deconv1"
        )
        self.deconv2 = tf.keras.layers.Conv2D(
            channel, kernel_size, padding='same', activation='relu', name="RelightNet_deconv2"
        )
        self.deconv3 = tf.keras.layers.Conv2D(
            channel, kernel_size, padding='same', activation='relu', name="RelightNet_deconv3"
        )
        self.feature_fusion = tf.keras.layers.Conv2D(
            channel, 1, padding='same', activation=None, name="RelightNet_feature_fusion"
        )
        self.output_layer = tf.keras.layers.Conv2D(
            1, 3, padding='same', activation=None, name="RelightNet_output"
        )

    def call(self, input_L, input_R):
        # 拼接输入
        input_im = concat([input_R, input_L])

        # 下采样路径
        conv0 = self.conv0(input_im)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # 上采样路径
        up1 = tf.image.resize(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]), method='nearest')
        deconv1 = self.deconv1(up1) + conv2
        up2 = tf.image.resize(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]), method='nearest')
        deconv2 = self.deconv2(up2) + conv1
        up3 = tf.image.resize(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]), method='nearest')
        deconv3 = self.deconv3(up3) + conv0

        # 特征融合
        deconv1_resize = tf.image.resize(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]), method='nearest')
        deconv2_resize = tf.image.resize(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]), method='nearest')
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])

        feature_fusion = self.feature_fusion(feature_gather)
        output = self.output_layer(feature_fusion)

        return output


class LowLightEnhance(tf.keras.Model):
    def __init__(self, checkpoint_dir):
        super(LowLightEnhance, self).__init__()
        self.DecomNet_layer_num = 5
        self.checkpoint_dir = checkpoint_dir
        # 定义输入
        self.input_low = tf.keras.Input(shape=(None, None, 3), name='input_low')
        self.input_high = tf.keras.Input(shape=(None, None, 3), name='input_high')
        # 创建 DecomNet 和 RelightNet
        self.DecomNet = DecomNet(layer_num=self.DecomNet_layer_num)
        self.RelightNet = RelightNet()
        # 为 DecomNet 和 RelightNet 创建独立的 Checkpoint 和 CheckpointManager
        self.DecomNet_checkpoint = tf.train.Checkpoint(DecomNet=self.DecomNet)
        self.RelightNet_checkpoint = tf.train.Checkpoint(RelightNet=self.RelightNet)
        # CheckpointManager 用于管理每个网络的保存和恢复
        self.DecomNet_checkpoint_manager = tf.train.CheckpointManager(self.DecomNet_checkpoint,
                                                                      checkpoint_dir + "/DecomNet", max_to_keep=5)
        self.RelightNet_checkpoint_manager = tf.train.CheckpointManager(self.RelightNet_checkpoint,
                                                                        checkpoint_dir + "/RelightNet", max_to_keep=5)
        # optimizer设置优化器
        self.DecomNet_optimizer = tf.keras.optimizers.Adam()
        self.RelightNet_optimizer = tf.keras.optimizers.Adam()
        print("[*] Initialize model successfully...")

    def call(self, input_low, input_high):
        # 如果 input_high 为 None，则使用与 input_low 相同形状的全零张量
        if input_high is None:
            input_high = tf.zeros_like(input_low)  # 或者你可以选择其他合适的默认值
        # 前向传播
        R_low, I_low = self.DecomNet(input_low)  # 对低光照图像使用DecomNet
        R_high, I_high = self.DecomNet(input_high)  # 对高光照图像使用DecomNet

        I_delta = self.RelightNet(I_low, R_low)

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])

        output_S = R_low * I_delta_3

        # 返回网络的输出
        return (R_low,I_low,R_high,I_high,I_delta,I_low_3,I_high_3,I_delta_3,output_S)


    def compute_loss_Decom(self, R_low, I_low_3, R_high, I_high_3, input_low, input_high,I_low,I_high):
        # DecomNet loss
        recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 - input_low))
        recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - input_high))
        recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - input_low))
        recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - input_high))
        equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))

        Ismooth_loss_low = self.smooth(I_low, R_low)
        Ismooth_loss_high = self.smooth(I_high, R_high)

        loss_Decom = recon_loss_low + recon_loss_high + 0.001 * recon_loss_mutal_low + 0.001 * recon_loss_mutal_high + \
                     0.1 * Ismooth_loss_low + 0.1 * Ismooth_loss_high + 0.01 * equal_R_loss

        return loss_Decom

    def compute_loss_Relight(self, R_low, I_delta_3, input_high,I_delta):
        # RelightNet loss
        relight_loss = tf.reduce_mean(tf.abs(R_low * I_delta_3 - input_high))
        Ismooth_loss_delta = self.smooth(I_delta, R_low)

        loss_Relight = relight_loss + 3 * Ismooth_loss_delta

        return loss_Relight


    def train_step(self, input_low, input_high, optimizer, phase, lr):
        optimizer.learning_rate.assign(tf.cast(lr, tf.float32))
        with tf.GradientTape() as tape:
            R_low, I_low, R_high, I_high, I_delta, I_low_3, I_high_3, I_delta_3, output_S = self.call(input_low, input_high)
            if phase == 'Decom':
                loss_Decom = self.compute_loss_Decom(R_low, I_low_3, R_high, I_high_3, input_low, input_high,I_low,I_high)
                total_loss = loss_Decom
                trainable_vars = self.DecomNet.trainable_variables
            elif phase == "Relight":
                loss_Relight = self.compute_loss_Relight(R_low, I_delta_3, input_high,I_delta)
                total_loss = loss_Relight
                trainable_vars = self.RelightNet.trainable_variables
            else:
                raise ValueError("Unknown task specified. Choose either 'DecomNet' or 'RelightNet'.")
        gradients = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        return total_loss

    def gradient(self, input_tensor, direction):
        # 定义平滑核，用于计算x方向和y方向的梯度
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        # 根据指定的方向选择相应的平滑核
        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y

        # 使用卷积计算梯度，并取绝对值
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding='SAME')(
            self.gradient(input_tensor, direction)
        )

    def smooth(self, input_I, input_R):
        # 将 input_R 转换为灰度图像
        input_R = tf.image.rgb_to_grayscale(input_R)

        # 计算平滑损失
        return tf.reduce_mean(
            self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) +
            self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y"))
        )

    def save(self, checkpoint, manager, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s at iteration %d" % (model_name, iter_num))
        save_path = manager.save(checkpoint_number=iter_num)  # Explicitly save with iter_num as the global step
        print(f"[*] Model saved at {save_path}")

    def load(self, checkpoint, manager, ckpt_dir):
        latest_checkpoint = manager.latest_checkpoint
        if latest_checkpoint:
            print(f"[*] Restoring model from {latest_checkpoint}")
            checkpoint.restore(latest_checkpoint).expect_partial()  # Restore from the latest checkpoint
            global_step = int(latest_checkpoint.split('-')[-1])  # Extract global step from checkpoint name
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def train(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, lr, sample_dir,
              ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)
        # load pretrained model
        if train_phase == "Decom":
            optimizer = self.DecomNet_optimizer
            checkpoint = self.DecomNet_checkpoint
            checkpoint_manager = self.DecomNet_checkpoint_manager
        elif train_phase == "Relight":
            optimizer = self.RelightNet_optimizer
            checkpoint = self.RelightNet_checkpoint
            checkpoint_manager = self.RelightNet_checkpoint_manager

        load_model_status, global_step = self.load(checkpoint=checkpoint, manager=checkpoint_manager,
                                                   ckpt_dir=ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (
        train_phase, start_epoch, iter_num))
        start_time = time.time()
        image_id = 0
        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)

                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(
                        train_low_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(
                        train_high_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)

                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data = zip(*tmp)

                # train
                loss = self.train_step(batch_input_low, batch_input_high, optimizer, train_phase, lr[epoch])

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(checkpoint, checkpoint_manager, iter_num, ckpt_dir, train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            output_R_low, output_I_low, output_I_delta, output_S = self.call(input_low_eval, None)
            if train_phase == "Decom":
                result_1 = output_R_low
                result_2 = output_I_low
            if train_phase == "Relight":
                result_1, result_2 = output_I_delta, output_S

            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1,result_2)

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag,ckpt_dir):
        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.DecomNet_checkpoint, self.DecomNet_checkpoint_manager,ckpt_dir=os.path.join(ckpt_dir, 'Decom'))
        load_model_status_Relight, _ = self.load(self.RelightNet_checkpoint,self.RelightNet_checkpoint_manager,ckpt_dir=os.path.join(ckpt_dir, 'Relight'))
        if load_model_status_Decom and load_model_status_Relight:
            print("[*] Load weights successfully...")

        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            R_low, I_low, R_high, I_high, I_delta, I_low_3, I_high_3, I_delta_3, output_S=self.call(input_low_test,None)

            if decom_flag == 1:
                print("hhhhhhhhhhhh,我老了")
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            save_images(os.path.join(save_dir, name + "_S." + suffix), output_S)