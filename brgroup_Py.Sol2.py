import pandas as pd
import tarfile
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_categorical_dtype




def main():
    # Set parameters
    pd.options.display.max_colwidth  = 500
    pd.options.display.float_format = '{:.0f}'.format
    input_data_path=os.path.abspath("data/input/dataset")
    output_data_path=os.path.abspath("data/output/")
    data_getter = DataGetter(input_data_path)
    visulizer = Vissulizer()
    data_getter.create_datasets()

    query_data=QueryData(data_getter)
    brgroup_case_2 = query_data.brgroup_case_2
    brgroup_case_2.columns = brgroup_case_2.columns.str.replace(' ', '')
    brgroup_case_2 = brgroup_case_2.dropna()
    print(brgroup_case_2.shape)
    # Sol 1 ----------------
    query_data.calculate_top_most_frequent_values(brgroup_case_2, 5)
    # sol 2
    query_data.search_typ_analysis(brgroup_case_2)
    # sol 3
    query_data.items_performance_analysis(brgroup_case_2, output_data_path, 10, 0.4)
    # Sol 4
    df = query_data.most_clicked_item_analysis(brgroup_case_2, output_data_path, 1000)
    #correlation value between CTR and avg_displayed_position for most 1000 clicked items
    corr = df['CTR'].corr(df['avg_displayed_position'])
    print(corr)
    # visulizer.frequency_distribution(df, 'CTR')
    # visulizer.frequency_distribution(df, 'avg_displayed_position')
    sorted_df = df.sort_values(by=['avg_displayed_position'], ascending=True).reset_index(drop=True)
    visulizer.plot(sorted_df, 'CTR', 'avg_displayed_position', 'CTR', 'Average Displayed Position'  )


class Vissulizer:

    def frequency_distribution(self, df:pd.DataFrame, col:str):
        """
        A user-defined function to plot histogram of input column of dataframe

        Parameters:
            df (pd.dataFrame): a dataFrame that is a source for calculating.
            col : column name to plot histogram
        Output:

        Returns:
            None
        """
        x = df[col][df[col] >= 0].to_numpy()
        ax = sns.kdeplot(x, shade=False, color='crimson')
        kdeline = ax.lines[0]
        mean = x.mean()
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        height = np.interp(mean, xs, ys)
        ax.vlines(mean, 0, height, color='crimson', ls=':')
        ax.fill_between(xs, 0, ys, facecolor='crimson', alpha=0.2)
        plt.show()

    def plot(self, df:pd.DataFrame, col1:str, col2:str, label1:str, label2:str):
        """
        A user-defined function to plot 2 input column of dataframe

        Parameters:
            df (pd.dataFrame): a dataFrame that is a source for calculating.
            col1 : column 1 name to plot
            col2 : column 2 name to plot
            label1 : label for column 1
            label2 : label for column 2
        Output:
            None
        Returns:
            None
        """
        plt.figure()
        x = df.index
        y1 = df[col1]
        y2 = df[col2]
        plt.plot(x,y1, label=label1)
        plt.plot(x,y2, label=label2)
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Magnitude', fontsize=12)
        plt.legend()
        plt.show()

class DataGetter:

    def __init__(self,input_data_path: str):
        self.input_data_path=input_data_path

    def unzip_data(self):
        '''DocString for this function'''
        with tarfile.open(self.input_zip_file, "r") as zip_ref:
            zip_ref.extractall("data/input/")
            zip_ref.close()


    def get_file(self,input_file : str)-> str:
        '''DocString for this function'''
        path = self.input_data_path
        for file in listdir(path):
            if join(path, file)==join(path,input_file):
                return file


    def data_into_df(self,input_file: str)-> pd.DataFrame:
        files = self.get_file(input_file)
        df = pd.read_csv(join(self.input_data_path,files))
        return df


    def create_datasets(self):
        brgroup_case_1 = self.data_into_df('data_analysis_case_study_part2.csv')
        return brgroup_case_1




class QueryData(DataGetter):
    """
    This is a class for querying data based on proposed KPIs .

    Attributes:
        dg (object): object of DataGetter class.
    """
    def __init__(self, dg):
        """
        The constructor for QueryData class.

        Parameters:
          dg (object): object of DataGetter class.
        """
        self.brgroup_case_2 = dg.create_datasets()

    def JoinDfs(self,leftDf:pd.DataFrame,rightDf:pd.DataFrame,leftkey:str,rightkey:str,howjoin:str)-> pd.DataFrame:
        """
        A user-defined function to execute join operation on given dataframes.

        Parameters:
            leftDf (pd.dataFrame): a dataFrame that is indicating a dataset on the left side of the join operation.
            rightDf (pd.dataFrame): a dataFrame that is indicating a dataset on the right side of the join operation.
            leftkey (string): a name of a column as a join key that belongs to a dataset on the left side of the join operation.
            rightkey (string): a name of a column as a join key that belongs to a dataset on the right side of the join operation.
            howjoin (string): a name of a type of join operation.

        Returns:
            (pd.dataFrame): a dataFrame that is a result of a merge operation on two given dataFrames.
        """
        return pd.merge(leftDf,rightDf,how=howjoin,left_on=leftkey,right_on=rightkey)

    def calculate_top_most_frequent_values(self,df:pd.DataFrame, num:int)  :
        """
        A user-defined function to calculate top most frequent values

        Parameters:
            df (pd.dataFrame): a dataFrame that is a source for calculating requested KPIs.
            num: Number of top most requested values per column


        Returns:
            None
         """
        for col in df.columns:
            print(col, end=' - \n')
            print('_' * 50)
            if  is_categorical_dtype(col):
                print(pd.DataFrame(df[col].astype('str').value_counts(dropna=False).sort_values(ascending=False).head(num)))
            else:
                print(pd.DataFrame(df[col].value_counts(dropna=False).sort_values(ascending=False).astype('str').head(num)))

    def search_typ_analysis(self,df:pd.DataFrame)  :
        """
        A user-defined function to analyse search types

        Parameters:
            df (pd.dataFrame): a dataFrame that is a source for calculating requested KPIs.


        Returns:
            None
         """
        # finding the best search_typecalculate top most frequent values
        brgroup_case_st_dp = df[['displayed_position', 'search_type']].groupby('search_type').agg(avg_displayed_position=('displayed_position', 'mean'))
        # finding the best sorting order for the best search_type
        brgroup_case_st_so = df[['search_type', 'sort_order']][df['search_type']==2116].groupby(['search_type', 'sort_order']).size().sort_values(ascending=False).reset_index(name='counts')
        # finding which search type should be excluded for a statistical reason
        brgroup_case_st = df[['displayed_position', 'search_type']].groupby('search_type').size().sort_values(ascending=False).reset_index(name='counts')
        print(brgroup_case_st_dp)
        print(brgroup_case_st_so)
        print(brgroup_case_st)


    def items_performance_analysis(self,df:pd.DataFrame, output_data_path:str, num: int, ctr_threshold: float)  :
        """
        A user-defined function to analyse top 10 “best” and “worst” performing items

        Parameters:
            df (pd.dataFrame): a dataFrame that is a source for calculating requested KPIs.
            output_data_path: output directory for explaind CSV
            num: Number of most requested records
            ctr_threshold: a threshold to determine best performing items by clicked_counts for CTR
        Output:
            output/output31.csv a csv containing calculate clicked_item_id,clicked_counts,impressed_item_ids,ipression_count,CTR items for all rows
            output/output32.csv a csv containing top 10 best performing items by clicked_counts and CTR>= ctr_threshold
            output/output33.csv a csv containing top 10 worst performing items by clicked_counts and CTR

        Returns:
            None
         """
        brgroup_case_2_impressed_item = df[['impressed_item_ids']].assign(impressed_item_ids = df.impressed_item_ids.str.split(';')).explode('impressed_item_ids')
        brgroup_case_2_impressed_item['impressed_item_ids'] = pd.to_numeric(brgroup_case_2_impressed_item['impressed_item_ids'])
        brgroup_case_2_impressed_item_cnt = brgroup_case_2_impressed_item.groupby(['impressed_item_ids'])['impressed_item_ids'].count().reset_index(name='ipression_count')
        brgroup_case_st = df[['clicked_item_id']].groupby('clicked_item_id').size().reset_index(name='clicked_counts')
        finalDF = self.JoinDfs(brgroup_case_st,brgroup_case_2_impressed_item_cnt, 'clicked_item_id', 'impressed_item_ids', 'inner' )
        finalDF['CTR'] = finalDF['clicked_counts'] / finalDF['ipression_count']
        finalDF.to_csv(output_data_path + '/output31.csv')
        sorted_df = finalDF[finalDF['CTR'] >= ctr_threshold ].sort_values(by=['clicked_counts', 'CTR'], ascending=False).head(num)
        sorted_df.to_csv(output_data_path + '/output32.csv')
        sorted_df_worst = finalDF.sort_values(by=['clicked_counts', 'CTR'], ascending=True).head(num)
        sorted_df_worst.to_csv(output_data_path + '/output33.csv')

    def most_clicked_item_analysis(self,df:pd.DataFrame, output_data_path:str, num: int)  :
        """
        A user-defined function calculate top 1000 most clicked items and CTR and average displayed position and correlation between average displayed position and correlation

        Parameters:
            df (pd.dataFrame): a dataFrame that is a source for calculating requested KPIs.
            output_data_path: output directory for CSV
            num: Number of top most requested values per column
        Output:
            output/output4.csv a csv containing calculated CTR, displayed position count, impressed item count and avg_displayed_position for most 1000 clicked items
        Returns:
            finalDF: a dataframe containing calculated CTR, displayed position count, impressed item count and avg_displayed_position for most 1000 clicked items
         """
        df1 = pd.DataFrame(df['clicked_item_id'].value_counts(dropna=False).sort_values(ascending=False).head(num))
        df1['index'] = df1.index
        brgroup_case_2_most1000 = df.loc[df['clicked_item_id'].isin(df1['index'])]
        brgroup_case_2_most1000 = brgroup_case_2_most1000.reset_index(drop=True)
        brgroup_case_2_most1000_without_11 = brgroup_case_2_most1000[['clicked_item_id', 'displayed_position']].loc[brgroup_case_2_most1000['displayed_position'] >= 0]
        brgroup_case_2_most1000_11 = brgroup_case_2_most1000[['user_id','session_id','clicked_item_id', 'displayed_position', 'impressed_item_ids']].loc[brgroup_case_2_most1000['displayed_position'] == -11]

        brgroup_case_21 = brgroup_case_2_most1000_11.assign(impressed_item_ids = brgroup_case_2_most1000_11.impressed_item_ids.str.split(';')).explode('impressed_item_ids')
        brgroup_case_21['impressed_item_ids'] = pd.to_numeric(brgroup_case_21['impressed_item_ids'])

        brgroup_case_21['rank'] = brgroup_case_21.groupby(['user_id','session_id', 'clicked_item_id']).cumcount()

        brgroup_case_21 = brgroup_case_21.reset_index(drop=True)
        # handle -11 in displayed_position with correct one
        brgroup_case_21.loc[((brgroup_case_21['displayed_position']  == -11.0) & (brgroup_case_21['clicked_item_id'] == brgroup_case_21['impressed_item_ids'].astype(np.float64) ) ), 'displayed_position'] = brgroup_case_21['rank']
        brgroup_case_21['displayed_position'] = brgroup_case_21.groupby(['user_id','session_id', 'clicked_item_id'])['displayed_position'].transform('max')
        # Select rank = 0 which select 1000 most clicked items with corrected values  of displayed_position for -11
        # Repalced by it's corresponding position in impressed_item_ids column
        brgroup_case_2_most1000_corr_11 = brgroup_case_21[['clicked_item_id', 'displayed_position', 'rank']][brgroup_case_21['rank'] == 0].reset_index(drop=True)
        brgroup_case_24 = pd.concat([brgroup_case_2_most1000_corr_11[['clicked_item_id', 'displayed_position']], brgroup_case_2_most1000_without_11]).reset_index(drop=True)
        brgroup_case_clicked_count_and_mean = brgroup_case_24.groupby(['clicked_item_id']).agg(cnt_displayed_position=('displayed_position', 'count'), avg_displayed_position=('displayed_position', 'mean'))

        brgroup_case_2_impressed_item = brgroup_case_2_most1000[['impressed_item_ids']].assign(impressed_item_ids = brgroup_case_2_most1000.impressed_item_ids.str.split(';')).explode('impressed_item_ids')
        brgroup_case_2_impressed_item['impressed_item_ids'] = pd.to_numeric(brgroup_case_2_impressed_item['impressed_item_ids'])
        brgroup_case_2_impressed_item_cnt = brgroup_case_2_impressed_item.groupby(['impressed_item_ids'])['impressed_item_ids'].count().reset_index(name='ipression_count')
        finalDF = self.JoinDfs(brgroup_case_clicked_count_and_mean,brgroup_case_2_impressed_item_cnt, 'clicked_item_id', 'impressed_item_ids', 'inner' )
        finalDF['CTR'] = finalDF['cnt_displayed_position'] / finalDF['ipression_count']
        finalDF.to_csv(output_data_path + '/output4.csv')

        return finalDF


if __name__ == "__main__":
    main()