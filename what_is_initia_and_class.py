class Tweet:
    def __init__(self, message, comma):
        self.message = message
        self.comma = comma

    def print_tweet(self):
        print('self,message')

    def for_fun(self):
        print('you are a bitch', self.comma, self.message)


Tweet('11', '89').for_fun()
