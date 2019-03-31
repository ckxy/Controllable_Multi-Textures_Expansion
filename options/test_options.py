from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--test_mode', type=str, default='w_test', help='test mode')
        self.parser.add_argument('--results_dir', type=str, default='/media/zhyang/Data/results/', help='saves results here.')
        self.parser.add_argument('--times', type=int, default=0, help='times of stress test')

        # vertical
        self.parser.add_argument('--which_epoch', type=str, default='', help='which epoch to load?')

        # horizontal
        self.parser.add_argument('--which_content', type=str, default='', help='which content image to load?')
        self.parser.add_argument('--which_style', type=str, default='', help='which style image to load?')

        # hidden
        self.parser.add_argument('--phase', type=int, default=0, help='name of folder')

        self.isTrain = False

