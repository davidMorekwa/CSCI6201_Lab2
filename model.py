from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def split_dataset(dataset):
    """
    Split dataset for training and testing
    """
    x = dataset.data
    y = dataset.target
    print(x.shape)
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, x_test, y_train, y_test


def train_model(dataset):
    """
    Train and test the model
    """
    x_train, x_test, y_train, y_test = split_dataset(dataset) 
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error: ", mse)