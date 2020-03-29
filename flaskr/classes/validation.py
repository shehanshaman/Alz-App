class ValidateUser:

    def has_data(user, columns):
        if user is None:
            heading = "Invalid User"
            content = "This user entry couldn't find."
            out = [heading, [content]]
            return out
        for col in columns:
            if user[col] is None:
                heading = "Data Not Found"
                content = col + " is None."
                out = [heading, [content]]

                return out

        return None

    def has_col(columns, features):
        columns = columns.to_list()
        if not ValidateUser.is_subset(columns, features):
            heading = "DataFrame Error"
            content = "Some features are not available in the dataframe."
            out = [heading, [content]]

            return out
        return None

    def is_subset(list1, list2):
        # True: list1 contains all elements in list2
        return all(elem in list1 for elem in list2)