import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df):
    # Churn distribution
    sns.countplot(x="Exited", data=df)
    plt.title("Customer Churn Distribution")
    plt.show()

    # Geography vs Churn
    sns.countplot(x="Geography", hue="Exited", data=df)
    plt.title("Churn by Geography")
    plt.show()

    # Age vs Churn
    sns.boxplot(x="Exited", y="Age", data=df)
    plt.title("Age vs Churn")
    plt.show()
