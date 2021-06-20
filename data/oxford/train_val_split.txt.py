import os
import csv


def read_train_val_split(txt_path):
    with open(txt_path) as f:
        sets = [x.rstrip() for x in f.readlines()]
    traversal_list = list(sets)
    return traversal_list

def read_tags_csv(csv_path):
    with open(csv_path) as csvfile:
        tags_reader = csv.reader(csvfile, delimiter=',')
        tags = []
        for row in tags_reader:
            tags += row
    return tags



if __name__ == '__main__':
    root = '/extssd/jiaxin/oxford'
    val_list = read_train_val_split(os.path.join(root, 'val.txt'))
    print('validation set list:')
    print(val_list)

    raw_traversal_list = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    raw_traversal_list.sort()

    train_list = []
    for traversal in raw_traversal_list:
        if traversal in val_list:
            print("Traversal %s in val set, skip." % traversal)
            continue
        if os.path.isfile(os.path.join(root, traversal, 'tags.csv')):
            tags = read_train_val_split(os.path.join(root, traversal, 'tags.csv'))
            if 'night' in tags:
                print("Traversal %s is night driving, skip." % traversal)
                continue
        else:
            print("Traversal %s is incomplete, skip." % traversal)
            continue


        train_list.append(traversal)

    with open(os.path.join(root, 'train.txt'), 'w') as f:
        f.writelines(["%s\n" % traversal for traversal in train_list])