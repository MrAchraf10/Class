from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

class WordCloudGenerator:
    def __init__(self, target, col, df):
        self.target = target
        self.col = col
        self.df = df
    
    def generate_wordclouds(self):
        for i in range(len(self.cl)):
            words_list = self.df.loc[self.df[self.target]==self.cl[i], self.col]
            words_str = ' '.join([str(word) for word in words_list])

            wordcloud = WordCloud(
                                  stopwords=STOPWORDS,
                                  background_color='white',
                                  width=800,
                                  height=400
                        ).generate(words_str)
            print(self.cl[i])
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.show()

        words_list = self.df[self.col]
        words_str = ' '.join([str(word) for word in words_list])

        wordcloud = WordCloud(
                              stopwords=STOPWORDS,
                              background_color='white',
                              width=800,
                              height=400
                    ).generate(words_str)
        print("ALL Classes:")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()

    def set_classes(self, cl):
        self.cl = cl