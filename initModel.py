from kaffe.tensorflow import Network


class ftRankLoss(Network):
    def setup(self):
        (self.feed('imgLow')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(512, name='fc8new'))

        (self.feed('fc7')
             .fc(256, name='fc8_BalancingElement')
             .fc(1, relu=False, name='fc9_BalancingElement'))

        (self.feed('fc7')
             .fc(256, name='fc8_ColorHarmony')
             .fc(1, relu=False, name='fc9_ColorHarmony'))

        (self.feed('fc7')
             .fc(256, name='fc8_Content')
             .fc(1, relu=False, name='fc9_Content'))

        (self.feed('fc7')
             .fc(256, name='fc8_DoF')
             .fc(1, relu=False, name='fc9_DoF'))

        (self.feed('fc7')
             .fc(256, name='fc8_Light')
             .fc(1, relu=False, name='fc9_Light'))

        (self.feed('fc7')
             .fc(256, name='fc8_MotionBlur')
             .fc(1, relu=False, name='fc9_MotionBlur'))

        (self.feed('fc7')
             .fc(256, name='fc8_Object')
             .fc(1, relu=False, name='fc9_Object'))

        (self.feed('fc7')
             .fc(256, name='fc8_Repetition')
             .fc(1, relu=False, name='fc9_Repetition'))

        (self.feed('fc7')
             .fc(256, name='fc8_RuleOfThirds')
             .fc(1, relu=False, name='fc9_RuleOfThirds'))

        (self.feed('fc7')
             .fc(256, name='fc8_Symmetry')
             .fc(1, relu=False, name='fc9_Symmetry'))

        (self.feed('fc7')
             .fc(256, name='fc8_VividColor')
             .fc(1, relu=False, name='fc9_VividColor'))

        (self.feed('fc8new', 
                   'fc8_BalancingElement', 
                   'fc8_ColorHarmony', 
                   'fc8_Content', 
                   'fc8_DoF', 
                   'fc8_Light', 
                   'fc8_MotionBlur', 
                   'fc8_Object', 
                   'fc8_Repetition', 
                   'fc8_RuleOfThirds', 
                   'fc8_Symmetry', 
                   'fc8_VividColor')
             .concat(-1, name='Concat9')
             .fc(128, name='fc10_merge')
             .fc(1, relu=False, name='fc11_score'))