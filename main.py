from setup import initialize_environment
from install import install_dependencies
from imports import *
from utils import *
from functions import *
from model import *

def main():

    initialize_environment()

    install_dependencies()

    print("Main workflow starts here.")

    channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] # Frontal = [0, 1, 2, 3, 4, 5], Central = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], Parietal = [18, 19, 20, 21],
    # All = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    # chan = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    task = 'left' # left, right, foot, tongue
    apply_filter = True
    time = [4] #[4, 2]
    band = [['a', 'a']] #[[0.5, 4], [4, 8], [8, 13], [13, 30], [30, 100], ['a', 'a']]
    num_epochs = 50

    for fl, fh in band:
        if fl == 'a':
            apply_filter = False
        else:
            # ------------------------------------------------------------------ Train and Validation Data -------------------------------------------------
            fs = 250  # Sampling frequency
            order = 5  # Filter order
            # Create bandpass filter coefficients
            nyq = 0.5 * fs
            low = fl / nyq
            high = fh / nyq
            b, a = butter(order, [low, high], btype='band')
        for t in time:
            df = []
            for i in range(1,10):
            data = loadmat(f'/gdrive/MyDrive/Motor_Imagery/BCI2a/subjects1000/sub{i}/data_{task}_sub{i}.mat')
            data_val = loadmat(f'/gdrive/MyDrive/Motor_Imagery/BCI2a/subjects1000_val/sub{i}/data_{task}_sub{i}.mat')
            if t == 4:
                data1 = data[f'data_{task}'][:,channels,:]
                data_val = data_val[f'data_{task}'][:,channels,:]
                data = np.concatenate((data1, data_val), axis=0)
                if apply_filter == True:
                data = filtfilt(b, a, data) #frequency filter
                label = [i for i in range(1, 10) for _ in range(data.shape[0])]
                label = np.array(label).reshape((9, data.shape[0]))
                df.append(data)
            if t == 2:
                data1 = data[f'data_{task}'][:,channels,:500]
                data2 = data[f'data_{task}'][:,channels,500:1000]
                data1_val = data_val[f'data_{task}'][:,channels,:500]
                data2_val = data_val[f'data_{task}'][:,channels,500:1000]
                data = np.concatenate((data1, data2, data1_val, data2_val), axis=0)
                if apply_filter == True:
                data = filtfilt(b, a, data) #frequency filter
                label = [i for i in range(1, 10) for _ in range(data.shape[0])]
                label = np.array(label).reshape((9, data.shape[0]))
                df.append(data)
            if t == 1:
                data1 = data[f'data_{task}'][:,channels,:250]
                data2 = data[f'data_{task}'][:,channels,250:500]
                data3 = data[f'data_{task}'][:,channels,500:750]
                data4 = data[f'data_{task}'][:,channels,750:1000]
                data1_val = data_val[f'data_{task}'][:,channels,:250]
                data2_val = data_val[f'data_{task}'][:,channels,250:500]
                data3_val = data_val[f'data_{task}'][:,channels,500:750]
                data4_val = data_val[f'data_{task}'][:,channels,750:1000]
                data = np.concatenate((data1, data2, data3, data4, data1_val, data2_val, data3_val, data4_val), axis=0)
                # data = np.concatenate((data1, data2, data3), axis=0) #for data with 75 sample
                if apply_filter == True:
                data = filtfilt(b, a, data) #frequency filter
                label = [i for i in range(1, 10) for _ in range(data.shape[0])]
                label = np.array(label).reshape((9, data.shape[0]))
                df.append(data)
            df = np.array(df)
            print(df.shape)
            num_trial = df.shape[1]
            num_ch = df.shape[2]
            num_smaple = df.shape[3]
            df = df.reshape((9*num_trial,num_ch,num_smaple))
            label = np.array(label)
            label = label.reshape((9*num_trial,))
            label = label -1
            x_train, x_valid, y_train, y_valid = train_test_split(df, label, test_size=0.2, random_state=23)
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=23)
            # print(x_train.shape, x_valid.shape, x_test.shape)
            # break
            x_train = torch.FloatTensor(x_train)
            x_train = x_train.unsqueeze(1)
            y_train = torch.LongTensor(y_train)
            y_train = y_train.squeeze()
            x_valid = torch.FloatTensor(x_valid)
            x_valid = x_valid.unsqueeze(1)
            y_valid = torch.LongTensor(y_valid)
            y_valid = y_valid.squeeze()
            x_test = torch.FloatTensor(x_test)
            x_test = x_test.unsqueeze(1)
            y_test = torch.LongTensor(y_test)
            y_test = y_test.squeeze()

            mu = x_train.mean(dim=0)
            std = x_train.std(dim=0)
            x_train = (x_train - mu) / std
            x_valid = (x_valid - mu) / std
            x_test = (x_test - mu) / std

            train_dataset = TensorDataset(x_train, y_train)
            valid_dataset = TensorDataset(x_valid, y_valid)
            test_dataset = TensorDataset(x_test, y_test)

            # --------------------------------------------------------------- K-Fold cross-validation -------------------------------------------------------
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            all_loss_test_hist = []
            all_acc_test_hist = []
            all_precision_test_hist = []
            all_recall_test_hist = []
            all_f1_test_hist = []
            all_loss_test_hist_s = []
            all_acc_test_hist_s = []
            all_precision_test_hist_s = []
            all_recall_test_hist_s = []
            all_f1_test_hist_s = []
            all_targests_test_hist = []
            all_outputs_test_hist = []

            for fold, (train_idx, valid_idx) in enumerate(kf.split(x_train)):
            print(f"Fold {fold+1}, fl = {fl}, t = {t}")
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)
            valid_loader = DataLoader(train_dataset, sampler=valid_sampler, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

            model = CNN().to(device)
            loss_fn = nn.MultiMarginLoss()
            lr = 0.00005
            wd = 3e-4
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            best_loss_valid = float('inf')

            for epoch in range(num_epochs):
                # Train
                model, loss_train, acc_train = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch)

                # Validation
                loss_valid, acc_valid, _, _ = validation(model, valid_loader, loss_fn)

                if loss_valid < best_loss_valid:
                    path = '/gdrive/MyDrive/Motor_Imagery'
                    torch.save(model, path + '/model_5_fold.pt')
                    best_loss_valid = loss_valid
                    print('Model Saved!')

                print(f'Valid: Loss = {loss_valid:.4}, Acc = {acc_valid:.4}')
                print()

            model = torch.load('/gdrive/MyDrive/Motor_Imagery/model_5_fold.pt')
            final_loss_test, final_acc_test, all_targets_test, all_outputs_test = validation(model, test_loader, loss_fn)
            acc, macro_precision, macro_recall, macro_f1 = cal_metrics(all_targets_test, all_outputs_test)

            all_loss_test_hist.append(final_loss_test)
            all_acc_test_hist.append(final_acc_test)
            all_precision_test_hist.append(macro_precision)
            all_recall_test_hist.append(macro_recall)
            all_f1_test_hist.append(macro_f1)

            #------------------------------------------------------------KD----------------------------------------------------------------------

            teacher = torch.load('/gdrive/MyDrive/Motor_Imagery/model_5_fold.pt')
            # teacher.eval()
            student = CNN().to(device)
            lr = 0.00005
            wd = 3e-4
            optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=wd)
            loss_fn = nn.MultiMarginLoss()

            best_loss_valid_s = torch.inf
            epoch_counter = 0


            for epoch in range(num_epochs):
                # Train
                student, loss_train, acc_train = train_one_epoch_kd(student,
                                                                    teacher,
                                                                    train_loader,
                                                                    loss_fn_kd,
                                                                    optimizer,
                                                                    epoch)
                # Validation
                loss_valid, acc_valid, _, _ = validation(student,
                                                        valid_loader,
                                                        loss_fn)

                if loss_valid < best_loss_valid_s:
                # path = '/gdrive/MyDrive/Motor_Imagery'
                # torch.save(model, path + '/model_6_fold.pt')
                best_loss_valid_s = loss_valid
                print('best')

                print(f'Valid: Loss = {loss_valid:.4}, Acc = {acc_valid:.4}')
                print()

                epoch_counter += 1

            # student = torch.load('/gdrive/MyDrive/Motor_Imagery/model_6_fold.pt')
            # student.eval()
            final_loss_test, final_acc_test, all_targets_test, all_outputs_test = validation(student, test_loader, loss_fn)
            acc, macro_precision, macro_recall, macro_f1 = cal_metrics(all_targets_test, all_outputs_test)

            all_loss_test_hist_s.append(final_loss_test)
            all_acc_test_hist_s.append(final_acc_test)
            all_precision_test_hist_s.append(macro_precision)
            all_recall_test_hist_s.append(macro_recall)
            all_f1_test_hist_s.append(macro_f1)
            all_targests_test_hist.append(all_targets_test)
            all_outputs_test_hist.append(all_outputs_test)

        # --------------------------------------------Save Results----------------------------------------------------------

            a1 = sum(all_loss_test_hist)/3
            a2 = sum(all_loss_test_hist_s)/3
            b1 = sum(all_acc_test_hist)/3
            b2 = sum(all_acc_test_hist_s)/3
            c1 = sum(all_precision_test_hist)/3
            c2 = sum(all_precision_test_hist_s)/3
            d1 = sum(all_recall_test_hist)/3
            d2 = sum(all_recall_test_hist_s)/3
            e1 = sum(all_f1_test_hist)/3
            e2 = sum(all_f1_test_hist_s)/3


            df = pd.DataFrame([[e1, d1, c1, b1*100, a1, e2, d2, c2, b2*100, a2]],
                            columns=['f1', 'recall', 'precision', 'acc', 'loss', 'f1_s', 'recall_s', 'precision_s', 'acc_s', 'loss_s'])

            # # Path to the Excel file
            # excel_file_path = '/gdrive/MyDrive/Motor_Imagery/resultsyyyyyyyy.xlsx'

            # if os.path.exists(excel_file_path):
            #     # If the file exists, read the existing data
            #     existing_df = pd.read_excel(excel_file_path)

            #     # Append the new data
            #     updated_df = pd.concat([existing_df, df], ignore_index=True)
            # else:
            #     # If the file does not exist, create a new DataFrame
            #     updated_df = df

            # # Write the updated DataFrame back to the Excel file
            # with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
            #     updated_df.to_excel(writer, index=False)

    all_targets_test_hists = np.concatenate([t.cpu().numpy() for t in all_targests_test_hist])
    all_outputs_test_hists = np.concatenate([t.cpu().numpy() for t in all_outputs_test_hist])

    # Now you can create the confusion matrix:
    cm = confusion_matrix(all_targets_test_hists, all_outputs_test_hists)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(9)
    plt.xticks(tick_marks, ['1','2','3','4','5','6','7','8','9'], rotation=45)
    plt.yticks(tick_marks, ['1','2','3','4','5','6','7','8','9'])
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == "__main__":
    main()
