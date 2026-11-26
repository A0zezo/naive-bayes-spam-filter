SPAM = ["win a free prize now", "free cash offer click now", "claim your free gift card today", "urgent you won money click here",
        "cheap loans approved instantly", "exclusive offer limited time free", "congratulations winner claim prize", "buy cheap pills online now",
        "earn money fast from home", "free entry win cash prize", "act now limited offer cash", "you have been selected for a free reward"]
HAM = ["are we still meeting for lunch tomorrow", "please review the project report today", "can you send me the meeting notes",
       "the deployment finished without errors", "lets schedule a call about the model", "thanks for the update on the dataset",
       "reminder team standup at ten", "i pushed the fix to the repository", "do you have time to review my pull request",
       "the training run completed overnight", "dinner at my place on friday", "attached is the draft of the paper"]
TRAIN = [(t, "spam") for t in SPAM[:9]] + [(t, "ham") for t in HAM[:9]]
TEST = [(t, "spam") for t in SPAM[9:]] + [(t, "ham") for t in HAM[9:]]
