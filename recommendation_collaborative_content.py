#----------------------------------------------------------------------------
# Created By  : Shruti Shrestha
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def findksimilarids(id, matrix, metric="cosine", k=10):
    """
    calculates the similarities and similar indices according to K nearest Neighbour algorithm
    :param id: id for which we find similar other items
    :param matrix: the matrix under consideration
    :param metric: type of correlation to perform
    :param k: the number of nearest neighbour
    :return: similarity values and similar indices
    """
    knn = NearestNeighbors(metric=metric, algorithm='brute')
    knn.fit(matrix)
    distances, indices = knn.kneighbors(matrix.iloc[id, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()
    return similarities, indices


class main():
    def __init__(self):
        """
        This function will initialize our data to self.data table, and initialize four other tables i.e user to item table, userToTimesVoted -> number of times the user has voted, itemTimesVoted -> number of times the items has been voted, and item_user_table
        """
        self.data = pd.read_csv("./take_home_ss_ratings.csv")
        self.user_item_table = self.userItemTable()
        self.user_to_time_voted = self.userToTimesVoted()
        self.item_times_voted = self.itemTimesVoted()
        self.item_user_table = self.user_item_table.T

    def userToTimesVoted(self):
        """
        counts the number of times the user has voted
        :returns: a dataframe with two columns named 'user_id', 'times_voted', which shows how many times the user has voted in total
        """
        user_to_time_voted = self.data['user_id'].value_counts().rename_axis('user_id').reset_index(name='times_voted')
        print(user_to_time_voted.columns)
        return user_to_time_voted

    def itemTimesVoted(self):
        """counts the number of times the item has been voted
        :returns: a dataframe which shows how many times the item has been voted
        """
        item_to_time_voted = self.data['item_id'].value_counts().rename_axis('item_id').reset_index(name='times_voted')
        return item_to_time_voted

    def userItemTable(self):
        """
        replaces disliked items rating from 0 to -1 and makes a pivot table by taking user_id and item_id columns
        :return: pivot column with user_id in rows and item_id in column, with 1 for liked item, -1 for disliked item and nan for items the user havent seen yet
        """
        data = self.data.copy()
        data['rating'][data['rating'] == 0] = -1
        self.user_item_table = pd.DataFrame(
            data.pivot_table(values='rating', index='user_id', columns='item_id', fill_value=np.nan))
        return self.user_item_table

    def popularItems(self):
        """
        calculates popular item by grouping the item_ids and counting their ratings
        :return: list of popular items of the website
        """
        popular_products = pd.DataFrame(self.data.groupby('item_id').sum()).sort_values('rating', ascending=False)
        return popular_products.index[:10].tolist()

    def itemVotedTheLeast(self):
        """
        calculates the items voted by the least users, grouped by the item_ids and counting their ratings
        :return: list of items voted less in the website
        """
        less_voted_products = pd.DataFrame(self.data.groupby('item_id').sum()).sort_values('rating', ascending=True)
        return less_voted_products.index[:10].tolist()

    def productsLikedByCurrentUser(self, user_id):
        """
        calculates the products liked by the user
        :param user_id: the used id whose details are to be operated
        :return: top 10 items liked by the user
        """
        one_user_data = self.user_item_table.loc[user_id]
        items_liked_by_user = one_user_data[one_user_data == 1]
        return items_liked_by_user.index[:10].tolist()

    def productsDislikedByCurrentUser(self, user_id):
        """
        calculates the item disliked by the user
        :param user_id: the used id whose details are to be operated
        :return: top 10 items disliked by the user
        """
        one_user_data = self.user_item_table.loc[user_id]
        items_disliked_by_user = one_user_data[one_user_data == -1]
        return items_disliked_by_user.index[:10].tolist()

    def answersAnweredOnAvergae(self):
        """
        calculates the average votes casted by the users by summing the counts of all users divided by the total number of user
        :return: scalar value representing the average vote casted by the user
        """
        return sum(self.data.groupby('user_id').count()["item_id"]) / self.data["user_id"].nunique()

    def userWhoVotedTheMost(self):
        """
        calculates the most active users by summing the number of votes cast by every individual user
        :return: top active users(most engaged in voting)
        """
        return self.user_to_time_voted["user_id"][:5].values

    def normalisedUserItemTable(self, user_item_table):
        """
        normalizes the user item table by substracting each value by the mean taken out by dividing the sum of rating casted by a user to the total vote casted by user
        :param user_item_table: table with user ids in row and item ids in column
        :return: normalized user item table with user ids as rows and item ids as columns
        """
        normalised_user_item_table = user_item_table.copy()

        for user_id in range(normalised_user_item_table.shape[0]):
            total_user_votes = self.user_to_time_voted.loc[self.user_to_time_voted['user_id'] == user_id]
            total_votes_casted_by_user = total_user_votes["times_voted"].values[0]
            mean = normalised_user_item_table.loc[user_id].sum() / total_votes_casted_by_user
            normalised_user_item_table.loc[user_id] = normalised_user_item_table.loc[user_id] - mean
        normalised_user_item_table.fillna(0, inplace=True)
        return normalised_user_item_table

    def normalisedItemUserTable(self, item_user_table):
        """
        normalizes the item user table by substracting each value by the mean taken out by dividing the sum of ratings casted to the item to the total vote casted to the item
        :param item_user_table: the transpose of user item table to take out the correlation between items,  item id in rows and user ids in column
        :return: normalized item user table with item as rows are user ids as columns
        """
        normalised_item_user_table = item_user_table.copy()

        for item_id in range(normalised_item_user_table.shape[0]):
            total_item_votes = self.item_times_voted.loc[self.item_times_voted['item_id'] == item_id]
            total_votes_casted_for_item = total_item_votes["times_voted"].values[0]
            mean = normalised_item_user_table.loc[item_id].sum() / total_votes_casted_for_item
            normalised_item_user_table.loc[item_id] = normalised_item_user_table.loc[item_id] - mean
        normalised_item_user_table.fillna(0, inplace=True)
        return normalised_item_user_table

    def findSimilarItemUsingUserToUserCollaborative(self, user_id, matrix, metric="cosine"):
        """
        calculates the similar ids by using the nearest neighbour algorithm using the cosine similarity
        :param user_id: the used id whose details are to be operated
        :param matrix: user item table, which has user on rows and items on column
        :param metric: states how the correlation is to be done
        :return: items ids liked by the users who are similar to user_id
        """

        user_item_table = self.normalisedUserItemTable(matrix)
        similarities, similar_user_indices = findksimilarids(user_id, user_item_table, metric=metric, k=10)
        similar_user_indices = similar_user_indices.flatten().tolist()
        similar_user_table = self.user_item_table.loc[similar_user_indices]
        similar_user_df = pd.DataFrame(similar_user_table.sum(axis=0), columns=["total_rating"])
        similar_users_items = similar_user_df[similar_user_df["total_rating"] > 0].sort_values(by='total_rating',
                                                                                               ascending=False)[:5]
        items_liked_by_similar_user = similar_users_items.index
        return items_liked_by_similar_user.tolist()

    def findSimilarItemUsingItemToItemCollaborative(self, top_items_liked_by_the_user, matrix, metric="cosine"):
        """
        calculates the similar item ids by using the nearest neighbour algorithm using the cosine similarity
        :param top_items_liked_by_the_user: items that are liked by the user
        :param matrix: item user table, which has item on rows and users on column
        :param metric: states how the correlation is to be done
        :return: items ids calculated by similarity with items liked by the user_id
        """
        item_user_table = self.normalisedItemUserTable(matrix)
        item_similar_to_that_liked_by_user = []
        for item in top_items_liked_by_the_user:
            similarities, similar_item_indices = findksimilarids(int(item), item_user_table, metric=metric, k=2)
            item_similar_to_that_liked_by_user += similar_item_indices.flatten().tolist()
        similar_items = list(set((set(item_similar_to_that_liked_by_user) - set(top_items_liked_by_the_user))))

        return similar_items

    def itemToRecommendToAUser(self, user_id):
        """
        sums up the total item id by popularity, user to user collaboration and item to item collaboration to recommend to a particular user
        :param user_id: the used id whose details are to be operated
        :return: items to recommend to the user having user_id
        """

        popular_item = self.popularItems()[:5]
        items_liked_by_similar_user = self.findSimilarItemUsingUserToUserCollaborative(user_id=user_id,
                                                                                       matrix=self.user_item_table,
                                                                                       metric="cosine")
        top_items_liked_by_the_user = self.productsLikedByCurrentUser(user_id)
        items_similar_to_that_item = self.findSimilarItemUsingItemToItemCollaborative(
            top_items_liked_by_the_user=top_items_liked_by_the_user, matrix=self.item_user_table, metric="cosine")
        dictionary = {"based_on_popularity:": popular_item,
                      "based on similar user-user collaborative filtering": items_liked_by_similar_user,
                      "based on similar item-item collaborative filtering": items_similar_to_that_item}
        return dictionary


def start(ss):

    x = input("\n\nPlease enter your choice from 1 to 7:\n")

    x = int(x)

    if x is 1:
        user_id = int(input("Enter a user between 0 to 19999, enter user id: "))
        if user_id >= 20000:
            return ("no user exists of this value")
        else:
            print("\n\n****** Items to recommend to a user : processing.....will take approx 50 sec ******\n")
            print(ss.itemToRecommendToAUser(user_id))
    if x is 2:
        print("\n\n****** The most popular items: ****** \n")
        print(ss.popularItems())
    if x is 3:
        print("\n\n****** Items voted the least:  ******\n")
        print(ss.itemVotedTheLeast())
    if x is 4:

        user_id = int(input("enter a user if between 0 to 19999, enter user id:"))
        if user_id >= 20000:
            return ("no user exists of this value")
        else:
            print("\n\n****** Top items liked by a user:  ******\n")
            print(ss.productsLikedByCurrentUser(user_id))
    if x is 5:
        user_id = int(input("enter a user if between 0 to 19999, enter user id:"))
        if user_id >= 20000:
            return ("no user exists of this value")
        else:
            print("\n\n****** Top items disliked by a user:  ******\n")
            print(ss.productsDislikedByCurrentUser(user_id))
    if x is 6:
        print("\n\n****** Avergae answers answered: ****** \n")
        print(ss.answersAnweredOnAvergae())
    elif x is 7:
        print("\n\n****** User who have voted the most/ active users:  ******\n")
        print(ss.userWhoVotedTheMost())
    start(ss)


if __name__ == "__main__":
    print("Wait for the content, currently initializing the program")

    ss = main()
    welcome_text = "Hi, what do you want to do today? \n 1. items to recommend to a user? \n2. get the most popular " \
                   "item \n 3. get the item voted the least? \n 4. top liked item by a user? \n 5. top disliked items " \
                   "by a user \n 6. how many answers are answered on an average \n 7.get the user who have voted the most " \
                   "most \n"
    print(welcome_text)
    start(ss)

