NUM_IMAGES = 10427

def idx_to_name(idx):
    return 'anime_sample_' + str(idx) + '.jpg'

if __name__ == '__main__':
    # Generate with a 80/20 split for now
    # nvm
    num_intervals = NUM_IMAGES // 3
    # split_idx = (num_intervals * 8) // 10
    split_idx = num_intervals - 100
    with open('./anime_sample_train.txt', 'w') as fin:
        for i in range(split_idx):
            fin.write(idx_to_name(3 * i) + ' ')
            fin.write(idx_to_name(3 * i + 2) + ' ')
            fin.write(idx_to_name(3 * i + 1) + '\n')
    with open('./anime_sample_test.txt', 'w') as fin:
        for i in range(split_idx, num_intervals):
            fin.write(idx_to_name(3 * i) + ' ')
            fin.write(idx_to_name(3 * i + 2) + ' ')
            fin.write(idx_to_name(3 * i + 1) + '\n')
