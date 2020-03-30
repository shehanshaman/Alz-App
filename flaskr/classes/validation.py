class ValidateUser:

    def has_data(user, columns):
        if user is None:
            heading = "Invalid User"
            content = "This user entry couldn't find."
            out = [heading, [content]]
            return out

        s = 0
        content_array = []
        for col in columns:
            if user[col] is None:
                s = 1
                content = col + " is None."
                content_array.append(content)

        heading = "Data Not Found"
        out = [heading, content_array]

        if s == 1:
            return out
        else:
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