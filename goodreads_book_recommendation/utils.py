def remove_book_outliers(books_df):
    outliers = [0, '0', 'DK Publishing Inc', 'Gallimard']
    for outlier in outliers:
        books_df.drop(books_df[
            books_df['Year-Of-Publication'] == outlier].index,
                   inplace=True)
