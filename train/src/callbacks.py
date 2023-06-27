def on_train_batch_end(trainer):
    with open("train_batches.txt", "w") as file:
        file.write("train batch end")


def on_val_batch_end(validator):
    with open("val_batches.txt", "w") as file:
        file.write("val batch end")
