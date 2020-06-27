# EmotionX-2019

## Abstract

Forked from [MeenaAlfons/EmotionX-2019](https://github.com/MeenaAlfons/EmotionX-2019) which is based on Bidirectional Encoder Representations from Transformers (BERT). The task includes two datasets with dialogues, only for Orignal Friends and EmotionPush. Each of the two datasets contains English-language dialogues. The order of the emotions in the "emotion" label is "neutral, joy, sadness, fear, anger, surprise, disgust". If the voting score is same such as "annotation":"2000201" among 5 voters. 

## Modify list

## preprocess_data.py
```
Line 144 preprocess_train_dev(friend_data_path, 'friends_train.json', output_dir, do_sanitize)
Line 145 preprocess_train_dev(emotionpush_data_path, 'emotionpush.train.json', output_dir, do_sanitize)
```

## processor.py 

Change 'augmented' = 0

```
    def save_dev(self, data_dir, dev_file, result_file, examples, preds):
        self.save(
            os.path.join(data_dir, dev_file),
            os.path.join(data_dir, result_file),
            examples,
            preds
            )
```

## Add 
## friends_majority.py 
## friends_majority_sol.py
## friends_others.py
## friends_others_sol.py

# How To Run

## Step 1. Install Dependencies

Use the following commands to install dependencies

```sh
git clone https://github.com/huggingface/pytorch-pretrained-BERT.git
cd pytorch-pretrained-BERT
git checkout master
python setup.py install
pip install ./
cd ..

pip install emoji

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..
```

## Clone the repository

```sh
git clone https://github.com/MeenaAlfons/EmotionX-2019.git
cd EmotionX-2019
```

## Step 2. Extract datasets

Extract the EmotionPush and Friends datasets in the following heirarchy:
```
_ . (current directory)
 |_ dataset
   |_ EmotionPush
   | |_ emotionpush.train.json
   | 
   |_ Friends
     |_ friends_train.json
``` 

You may download the Friends dataset through this following link:
```
http://doraemon.iis.sinica.edu.tw/emotionlines/download.html
```

## Step3. Run Preprocessing

```
python EmotionX-2019/preprocess_data.py ./dataset
```

## Step 4. Train Majority Processor (Binary Classification)

```
python EmotionX-2019/friends_majority.py
```

## Step 5. Train Others Processor (Multi Classification)

```
python EmotionX-2019/friends_others.py
```

## Step 6. Test Majority Processor 

Classify whether one dialogue is "neutral" or "$Blank" (maybe one of the 7 labels)

```
python EmotionX-2019/friends_majority_sol.py
```

## Step 7. Test Others processor

Classify one dialogue's emotion label among the joy, sadness, fear, anger, surprise, disgust, or non-neutral based on the result of step 6's majority processor

```
python EmotionX-2019/friends_others_sol.py
```
