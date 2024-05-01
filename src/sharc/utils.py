class SharcLabel:
    NO = -1
    NOT_ENOUGH_INFO = 0
    YES = 1


def get_sharc_label(label):
    if label == "Yes":
        return SharcLabel.YES
    elif label == "No":
        return SharcLabel.NO
    else:
        return SharcLabel.NOT_ENOUGH_INFO

def create_conv_history(history):
    conv = ""
    for x in history:
        conv += f"Q: {x['follow_up_question']}\n"
        conv += f"A: {x['follow_up_answer']}\n"
    # remove last newline
    conv = conv[:-1]
    return conv