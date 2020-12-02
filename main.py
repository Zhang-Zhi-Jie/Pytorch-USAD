
def train(train_dataLoader, model, criterion, train_epochs, device):
    for epoch in range(train_epochs):
        for x in train_dataLoader:
            z, x_hat1, x_hat2, x_hat12 = model(x)
            l_ae1 = 1/n * criterion(x, x_hat1) + (1 - 1/n) * criterion(x, x_hat12)
            l_ae2 = 1/n * criterion(x, x_hat2) - (1 - 1/n) * criterion(x, x_hat12)


def load_data(data_type, w_size, s_size):
    if data_type == "SMD":
        train_path = "./data/SMD/processed/machine-1-1_train.pkl"
        test_path = "./data/SMD/processed/machine-1-1_test.pkl"
        train_x = read_pkl(train_path)
        train_y = np.zeros((train_x.shape[0], 1))
        test_x = read_pkl(test_path)
        test_y = read_pkl("./data/SMD/processed/machine-1-1_test_label.pkl")
        train_x = min_max_scaling(train_x)
        input_dim = train_x.shape[1]
        test_x = min_max_scaling(test_x)
    elif data_type == "KDD":
        train_path = "./data/KDD/kdd99_train.npy"
        test_path = "./data/KDD/kdd99_test.npy"
        train_data = read_npy(train_path)
        test_data = read_npy(test_path)
        train_x, train_y = train_data[:,:-1], train_data[:,-1]
        test_x, test_y = test_data[:,:-1], test_data[:,-1]
        train_x = min_max_scaling(train_x)
        input_dim = train_x.shape[1]
        test_x = min_max_scaling(test_x)
    s_train_x, s_train_y= split_data(train_x, train_y, w_size, s_size)
    train_loader = data_to_dataLoader(s_train_x, s_train_y, 1, True)
    s_test_x, s_test_y= split_data(test_x, test_y, w_size, s_size)
    test_loader = data_to_dataLoader(s_test_x, s_test_y, 1, False)
    data_loader = {"train":train_loader, "test":test_loader}
    return data_loader, input_dim

if __name__ == '__main__':

            