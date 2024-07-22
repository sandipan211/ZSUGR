import random
import pandas as pd

# label id to label name mapping 
label_dict =    {
    # -1: "negative",
    0:  "start_comm",
    1:  "end_comm",
    2:  "up",
    3:  "down",
    4:  "photo",
    5:  "backwards",
    6:  "carry",
    7:  "boat",
    8:  "here",
    9:  "mosaic",
    10: "num_delimiter",
    11: "one",
    12: "two",
    13: "three",
    14: "four",
    15: "five"
}

def make_caddy_splits(train_path, test_seen_path, test_unseen_path, args):

    cols = ['scenario','stereo left','stereo right','label name','label id','synthetic']
    file1_path = args.root + '/' + args.image_folder + '/CADDY_gestures_all_true_positives_release_v2.csv'
    df_p = pd.read_csv(file1_path)
    df_p = df_p[cols]
    df_p = df_p[df_p['synthetic'] == 0]

    file2_path = args.root + '/' + args.image_folder + '/CADDY_gestures_all_true_negatives_16_release_v2.csv'
    df_n = pd.read_csv(file2_path)
    df_n = df_n[cols]
    df_n = df_n[df_n['synthetic'] == 0]

    total_df = pd.concat([df_n, df_p], ignore_index=True)
    # Count unique values in each column
    unique_counts = total_df.nunique()

    # new change
    # print(df_p.nunique())
    # print(df_p.groupby('label id').count()*2)
    # # exit(0)

    # # Display the unique counts
    # print(f'Total raw images: {unique_counts}')

    # output_file_path = '/content/drive/MyDrive/MTP_caddy/total_raw_images.csv'

    # # Write the DataFrame to a CSV file
    # total_df.to_csv(output_file_path, index=False)

    left_df= total_df.drop("stereo right",axis=1)
    right_df= total_df.drop("stereo left",axis=1)
    left_df.rename(columns = {'stereo left':'image_path'}, inplace = True)
    right_df.rename(columns = {'stereo right':'image_path'}, inplace = True)
    common_df = pd.concat([left_df, right_df], ignore_index=True)
    image_label_df=common_df[['image_path','label id']]
    # output_file_path = '/content/drive/MyDrive/MTP_caddy/t_raw_images_label.csv'

    # Write the DataFrame to a CSV file
    # image_label_df.to_csv(output_file_path, index=False)
    # Specify custom label IDs for the two DataFrames

    if args.split_type == 'random':
        sample_num = int(args.num_classes/2) - 2
        # unseen_label_ids = sorted(random.sample([label for label in label_dict.keys() if label != -1], sample_num))
        unseen_label_ids = sorted(random.sample([label for label in label_dict.keys()], sample_num))
        seen_label_ids = list(set(label_dict.keys()) - set(unseen_label_ids))

    elif args.split_type == 'RF' or args.split_type == 'NF':
        image_label_df = image_label_df[image_label_df['label id'] != 16]
        label_counts = image_label_df.groupby('label id').size()
        # Get the maximum count
        max_count = label_counts.max()
        print("Maximum count among label ids: \n", max_count)
        threshold = 0.25 * max_count

        # Initialize lists to store labels
        labels_above_threshold = []
        labels_below_threshold = []

        # Iterate over label counts
        for label, count in label_counts.items():
            if count >= threshold:
                labels_above_threshold.append(label)
            else:
                labels_below_threshold.append(label)

        print("Labels with count >= 25% of max count: \n", labels_above_threshold)
        print("Labels with count < 25% of max count: \n", labels_below_threshold)
        if args.split_type == 'RF':
            seen_label_ids = labels_above_threshold
            unseen_label_ids = labels_below_threshold
        elif args.split_type == 'NF':
            seen_label_ids = labels_below_threshold
            unseen_label_ids = labels_above_threshold

    print("Unseen labels: ",unseen_label_ids)
    print("Seen labels: ",seen_label_ids)

    # Create two DataFrames based on custom label IDs
    df_seen = image_label_df[image_label_df['label id'].isin(seen_label_ids)]
    df_test_seen = df_seen.groupby('label id').apply(lambda x: x.sample(frac=0.1))
    df_train = pd.concat([df_seen, df_test_seen]).drop_duplicates(keep=False)
    # Reset the index of the sampled DataFrame
    df_test_seen.reset_index(drop=True, inplace=True)
    # Reset the index of the remaining DataFrame
    df_train.reset_index(drop=True, inplace=True)
    df_test_unseen = image_label_df[image_label_df['label id'].isin(unseen_label_ids)]
    # Count unique values in each column
    unique_counts = df_seen.nunique()

    # Display the unique counts
    print(f'Number of seen: {unique_counts}')
    # Count unique values in each column
    unique_counts = df_test_unseen.nunique()

    # Display the unique counts
    print(f'Number of unseen: {unique_counts}')

    # Write the DataFrame to CSV files
    df_train.to_csv(train_path, index=False)
    df_test_seen.to_csv(test_seen_path, index=False)
    df_test_unseen.to_csv(test_unseen_path, index=False)

    return train_path, test_seen_path, test_unseen_path

