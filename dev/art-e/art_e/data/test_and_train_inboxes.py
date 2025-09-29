import random

# Randomly chosen from recipients with at least 5000 emails
test_inboxes = [
    "karen.denne@enron.com",
    "john.lavorato@enron.com",
    "paul.kaufman@enron.com",
    "sarah.novosel@enron.com",
    "greg.whalley@enron.com",
    "james.steffes@enron.com",
    "sara.shackleton@enron.com",
    "tim.belden@enron.com",
]

# Randomly chosen from recipients with at least 5000 emails
train_inboxes = [
    "louise.kitchen@enron.com",
    "gerald.nemec@enron.com",
    "susan.mara@enron.com",
    "alan.comnes@enron.com",
    "joe.hartsoe@enron.com",
    "william.bradford@enron.com",
    "harry.kingerski@enron.com",
    "kate.symes@enron.com",
    "richard.shapiro@enron.com",
    "sally.beck@enron.com",
    "jeff.dasovich@enron.com",
    "kay.mann@enron.com",
    "daren.farmer@enron.com",
    "steven.kean@enron.com",
    "richard.sanders@enron.com",
    "mark.taylor@enron.com",
    "pete.davis@enron.com",
    "elizabeth.sager@enron.com",
    "tana.jones@enron.com",
    "sandra.mccubbin@enron.com",
]


def get_inbox(split="test") -> str:
    if split == "test":
        return random.choice(test_inboxes)  # type: ignore
    else:
        return random.choice(train_inboxes)  # type: ignore
