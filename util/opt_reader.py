from collections import OrderedDict


def opt_reader(opt_dir, para_name, para_type):
    od = OrderedDict([])
    with open(opt_dir, 'r') as file:
        for line in file:
            line_list = line.split(':')
            if len(line_list) > 1:
                for i in range(len(para_name)):
                    if para_name[i] == line_list[0]:
                        data = line_list[1].lstrip().replace('\n', '')
                        if para_type[i] == 'int':
                            od[para_name[i]] = int(data)
                        elif para_type[i] == 'str':
                            od[para_name[i]] = data
                        elif para_type[i] == 'bool':
                            if data == 'True':
                                od[para_name[i]] = True
                            else:
                                od[para_name[i]] = False

    return od
