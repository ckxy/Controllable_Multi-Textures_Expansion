import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

model = create_model(opt)
TestOptions().printandsave(opt)
t1, t2 = model.get_titles()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, t1)
webpage = html.HTML(web_dir, t2)

for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()


