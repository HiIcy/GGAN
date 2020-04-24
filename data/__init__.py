pic=r"F:\Resources\DataSets\Facade\facade\test_picture"
lab=r"F:\Resources\DataSets\Facade\facade\test_label"
base=r'F:\Resources\DataSets\Facade\facade'
def split(mudi):
    from pathlib import Path
    import os
    import shutil
    pics = os.listdir(mudi)
    pics = [os.path.join(mudi,i) for i in pics]
    nums = len(pics)
    train = pics[:int(nums*0.1)]
    val = pics[int(nums*0.1):int(nums*0.2)]
    test = pics[:]

    train_fine_dir = Path(base)/"test_picture_fine"
    if not train_fine_dir.exists():
        train_fine_dir.mkdir()
    train_fine_train_dir=train_fine_dir/"train"
    train_fine_val_dir=train_fine_dir/"val"
    train_fine_test_dir=train_fine_dir/"test"
    if not train_fine_train_dir.exists():
        train_fine_train_dir.mkdir()
    if not train_fine_val_dir.exists():
        train_fine_val_dir.mkdir()
    if not train_fine_test_dir.exists():
        train_fine_test_dir.mkdir()
    for tr in train:
        shutil.copy2(tr,str(train_fine_train_dir))
    for tr in val:
        shutil.copy2(tr,str(train_fine_val_dir))
    for tr in test:
        shutil.copy2(tr,str(train_fine_test_dir))
# split(pic)