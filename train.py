from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import os
from tqdm import tqdm

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
TrainOptions().printandsave(opt)
print('#training images = %d' % dataset_size)
print('#len of dataset = %d' % len(dataset))

model = create_model(opt)
visualizer = Visualizer(opt)
visualizer.set_x(opt, len(dataset))
total_steps = 0

opt.results_dir = os.path.join(os.path.dirname(opt.checkpoints_dir), 'results')
web_dir = os.path.join(opt.results_dir, opt.name)
webpage = html.HTML(web_dir, 'Experiment = %s' % (opt.name))

for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):

    for i, data in enumerate(dataset):
        if epoch == opt.epoch_count and i == 0:
            save_data = data
        else:
            save_data = save_data

        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, i, errors)
            if total_steps % (2 * opt.print_freq) == 0:
                visualizer.display_errors_plot()

        total_steps += 1

        if total_steps % opt.display_freq == 0:
            model.set_input(save_data)
            model.forward()
            visuals = model.get_current_visuals()
            visualizer.display_current_results(visuals, epoch, i)

    if epoch % opt.save_epoch_freq == 0:
        model.save(epoch)

    if epoch > opt.niter:
        model.update_learning_rate()

visualizer.display_errors_plot()
print('Finished')
