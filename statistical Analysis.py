import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import inspect
import os.path

# get current dir
def get_file_path():
    """
    This function returns the current path directory
    """
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    return path


# This takes the world data as input path
input_path = str(get_file_path())+r"\world_data.csv"

# Load your CSV file (Replace 'your_file_path' with the actual file path)
data = pd.read_csv(input_path)


# Define the filtered_read_transpose function
def filtered_read_transpose(data):
    """
    Defines the data transformation from original data
    
    parameters:
        
        data: The original data in the form of pandas dataframe.
        
   Returns:  
        
       data_countries: duplicate dataframe with indicator column dropped and NaN-filtered rows.
       data_years: A transposed datframe which represents the data for years
       
    """
    # Remove rows with NaN values from the original data
    df_transformed = data.dropna()
    # Creating a copy of the cleaned data
    df_copy = df_transformed.copy()
    # Dropping the 'Indicator' column
    data_countries = df_copy.drop(["Indicator"], axis=1)
    # Transposing the data for years
    data_years = data_countries.transpose()

    return data_countries, data_years

 # clean the transposed dataframe by handling missing values
    cleaned_data_years = clean_transposed_data(data_years)

    return data_countries, cleaned_data_years

# Define the function to clean the transposed data
def clean_transposed_data(transposed_data):
    """
      The transposed dataframe was cleaned by handling missing values
      
    parameters:
        The transposed dataframe in the form of pandas datframe

    Returns:
        which returns the cleaned transposed dataframe
    """
    # Handling missing values by forward filling along each row
    cleaned_data = transposed_data.fillna(method='ffill', axis=1)
    
    # Additional cleaning steps can be added based on specific requirements
    
    return cleaned_data


# Process the loaded data using the defined function
data_countries, data_years = filtered_read_transpose(data)

# Displaying the transformed data
print("Data by Countries:")
print(data_countries.head())

print("\nData by Years (Transposed and Null Values Removed):")
print(data_years.head())


#Defines the attributes of the pie chart to design 
def design_pie_chart(data, year, explode_factor=0.2, colors=None, fontsize=12, autopct='%1.2f%%'):
    # Create a figure with subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Adding explode to separate slices
    explode = [explode_factor] * len(data)
    
    # Create a pie chart with adjusted start angle and counterclockwise direction
    wedges, texts, autotexts = ax.pie(
        data[year],
        labels=data['Country Name'],
        autopct=autopct,
        startangle=90,  # Adjust the start angle
        counterclock=False,  # Change the direction to clockwise
        colors=colors,  # Use custom colors if provided
        explode=explode,  # Separate slices
        wedgeprops={"linewidth": 1, "antialiased": False }  # Enhanced aesthetics
    )
    

    # Set font size for labels
    plt.setp(texts, fontsize=fontsize)
    plt.setp(autotexts, fontsize=fontsize, color="white")

    # Set a title
    ax.set_title(f'Pie Chart of Cereal Yield for Year {year}', fontsize=14)

    # Display the pie chart
    plt.show
    
# Example usage
preferred_countries = ['Argentina', 'Brazil', 'Australia', 'Bangladesh','United Kingdom']
year = '1990'
chosen_indicator = ["Rural land area where elevation is below 5 meters (sq. km)"]


# Filtering the data for the specified countries and year
data_filtered = data[(data['Country Name'].isin(preferred_countries)) & (data['Indicator'].isin(chosen_indicator))]
data_pie = data_filtered[['Country Name', year]].dropna()

# Generating the designed pie chart with the actual data
design_pie_chart(data_pie, year, explode_factor=0.1, autopct='%1.1f%%', fontsize=12)

#To analyse and display the overall statiscs of filtered data
compute_statistics = data_filtered.describe()
print(compute_statistics)



# defines the comparision among multiple countries with different indicators
def heatmap(data, countries, year, indicators):
    # Filter data for the specified countries, year, and indicators
    data_filtered = data[(data['Country Name'].isin(countries)) & (data['Indicator'].isin(indicators))]
    
    # Pivot the data to create a correlation matrix
    data_pivot = data_filtered.pivot(index='Country Name', columns='Indicator', values=year)
    
    # Calculate the correlation matrix
    correlation_matrix = data_pivot.corr()

    # Create a heatmap
    plt.figure(figsize=(12, 10))
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    
    # Add indicator names as x and y tick labels
    plt.xticks(np.arange(len(indicators)), indicators, rotation=90)
    plt.yticks(np.arange(len(indicators)), indicators)
    
    # Loop through each cell and add correlation values
    for i in range(len(indicators)):
        for j in range(len(indicators)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='w')
    
    # Set axis labels and title
    plt.xlabel('Indicators')
    plt.ylabel('Indicators')
    plt.title('Correlation Heatmap of Indicators')
    
    # Customize the colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # Display the heatmap
    plt.show()


countries_heatmap = ['Arab World', 'China', 'Spain', 'India', 'Netherlands']
year_int = '2015'
indicators_heatmap = [
    'Rural land area where elevation is below 5 meters (sq. km)',
    'Urban population',
    'Cereal yield (kg per hectare)',
    "Forest area (% of land area)",
    'Urban land area where elevation is below 5 meters (sq. km)']

# Correlation Heatmap of various indicators
heatmap(data, countries_heatmap, year_int, indicators_heatmap)




def bar_plot(data, countries, indicator_name, years):   
    data_filtered = data[(data['Country Name'].isin(countries)) & (data['Indicator'].isin(indicator_name))][['Country Name'] + years].fillna(0)

    data_prepared = data_filtered.melt(id_vars=['Country Name'], var_name='Year', value_name=indicator_name[0])

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid", palette="bright")
    sns.barplot(x='Year', y=indicator_name[0], hue='Country Name', data=data_prepared)

    plt.title(f'{indicator_name[0]} ({years[0]}-{years[-1]})')
    plt.xlabel('Year')
    plt.ylabel(f'{indicator_name[0]} in sq. km')  # Update the y-axis label

    plt.xticks(rotation=45)
    plt.ticklabel_format(style='plain', axis='y')

    plt.legend(title='', loc="upper left")

    plt.show()
    return

# Specified years, countries, indicators for the Bar Plot
selected_years = ['1990', '2000', '2015']
selected_countries = ['Arab World', 'China', 'Spain', 'India', 'Netherlands']
selected_indicator = ["Urban land area where elevation is below 5 meters (sq. km)"]

# Calling the function to plot the data for Bar Plot
bar_plot(data, selected_countries, selected_indicator, selected_years)




def power_bar_plot(data, countries, indicator_name, years):
    # Convert years to integers
    years_int = [int(year) for year in years]
    
    # Filter data for the selected countries, indicator, and years
    data_filtered = data[(data['Country Name'].isin(countries)) & (data['Indicator'].isin(indicator_name))][['Country Name'] + years].fillna(0)

    # Melting the DataFrame to make it suitable for Seaborn barplot
    data_melted = data_filtered.melt(id_vars=['Country Name'], var_name='Year', value_name=indicator_name[0])

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    sns.barplot(x='Year', y=indicator_name[0], hue='Country Name', data=data_melted)
    plt.title('Rural land for selected countries')
    plt.xlabel('Year')
    plt.ylabel(f'{indicator_name[0]}')
    plt.xticks(rotation=45)
    plt.legend(title='Country Name', loc="upper right")
    plt.show()

# Specified years, countries, and indicator for the Power Bar Plot
selected_years = ['1990', '2000', '2012','2015']
selected_countries = ['Argentina', 'Brazil', 'Australia', 'Bangladesh', 'United Kingdom']
selected_indicator = ["Rural land area where elevation is below 5 meters (sq. km)"]
print(data)
# Calling the function to plot the data for power bar plot style
power_bar_plot(data, selected_countries, selected_indicator, selected_years)



def scatter_plot(data, countries, indicator_name, years_int):
    # Filtering the data for the selected countries and years
    data_filtered = data[(data['Country Name'].isin(countries)) & (data['Indicator'].isin(indicator_name))][['Country Name'] + years_int].fillna(0)

    # Melting the DataFrame to make it suitable for Seaborn scatter plot
    data_melted = data_filtered.melt(id_vars=['Country Name'], var_name='Year', value_name='Forest area (% of land area)')

    # Plotting a scatter plot with dotted lines
    plt.figure(figsize=(12, 6))
    sns.set(style='whitegrid', palette='husl')
    sns.scatterplot(x='Year', y='Forest area (% of land area)', hue='Country Name', style='Country Name', markers=True, data=data_melted)
    sns.lineplot(x='Year', y='Forest area (% of land area)', hue='Country Name', style='Country Name', markers=False, dashes=True, data=data_melted)
    plt.title('Forest area (% of land area)')
    plt.xlabel('Year')
    plt.ylabel('Forest area (% of land area)')
    plt.xticks(rotation=45)
    plt.legend(title='Country Name', loc="upper left")
    plt.show()
    return

# Specified years, countries, indicators for the Scatter Plot
years_int = ['1990', '2000', '2015']
selected_countries = ['Arab World', 'China', 'Spain', 'India', 'Netherlands']
selected_indicator = ["Forest area (% of land area)"]

# Calling the function to plot the data for Scatter Plot with dotted lines
scatter_plot(data, selected_countries, selected_indicator, years_int)




def histogram(data, countries, indicator_name, years_int):
    # Filtering the data for the selected countries and years
    data_filtered = data[(data['Country Name'].isin(countries)) & (data['Indicator'].isin(indicator_name))][['Country Name'] + years_int]

    # Melting the DataFrame to make it suitable for Seaborn histogram
    data_melted = data_filtered.melt(id_vars=['Country Name'], var_name='Year', value_name=f'Cereal yield (kg per hectare)')

    # Create a colorful palette
    colorful_palette = sns.color_palette("husl", len(countries))

    # Plotting with histogram
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    for i, country in enumerate(countries):
        sns.histplot(data_melted[data_melted['Country Name'] == country], x=f'Cereal yield (kg per hectare)', bins=20, kde=True, label=country, color=colorful_palette[i])

    plt.title(f'Distribution of Cereal Yield (kg per hectare)')
    plt.xlabel(f'Cereal yield (kg per hectare)')
    plt.ylabel('Frequency')
    plt.legend(title='Country Name', loc="upper right")
    plt.show()
    return

# Specified years, countries, indicators for the Histogram
years_for_histogram = ['1990', '2000', '2015']
selected_countries_histogram = ['Arab World', 'China', 'Spain', 'India', 'Netherlands']
selected_indicator_histogram = ["Cereal yield (kg per hectare)"]

# Calling the function to plot the data for Histogram
histogram(data, selected_countries_histogram, selected_indicator_histogram, years_for_histogram)




def line_plot(data, countries, indicator_name, years_int):
    # Filtering the data for the selected countries and years
    data_filtered = data[(data['Country Name'].isin(countries)) & (data['Indicator'].isin(indicator_name))][['Country Name'] + years_int].fillna(0)

    # Melting the DataFrame to make it suitable for Seaborn lineplot
    data_melted = data_filtered.melt(id_vars=['Country Name'], var_name='Year', value_name='Urban population')

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid", palette="husl")
    sns.lineplot(x='Year', y='Urban population', hue='Country Name', data=data_melted)
    plt.title('Urban population')
    plt.xlabel('Year')
    plt.ylabel('Urban population')
    plt.xticks(rotation=45)
    plt.legend(title='Country Name', loc="upper left")
    plt.show()
    return

# Specified years, countries, indicators for the Line Plot
years_int = ['1990', '2000', '2015']
selected_countries = ['Arab World', 'China', 'Spain', 'India', 'Netherlands']
selected_indicator = ["Urban population"]

# Calling the function to plot the data
line_plot(data, selected_countries, selected_indicator, years_int)



#Defines the measuring of skewness of  data distribution whether it is positive or negative
def skew(data, countries, indicator_name, years):
    """Calculates the centralized and normalized skewness of distribution."""
    # Filter data
    data_filtered = data[(data['Country Name'].isin(countries)) & (data['Indicator'].isin(indicator_name))][['Country Name'] + years].fillna(0)

    # Prepare the DataFrame for skewness calculation
    data_prepared = data_filtered.melt(id_vars=['Country Name'], var_name='Year', value_name='Emissions')
    dist = data_prepared["Emissions"]

    # Calculate average and standard deviation for centralizing and normalizing
    aver = np.mean(dist)
    std = np.std(dist)

    # Calculate skewness
    value = np.sum(((dist - aver) / std)**3) / len(dist)
    print("Skewness =", np.round(value, 6))
    return value

#Defines the kurtosis of data which is heavier or lighter tailed of distribution
def kurtosis(data, countries, indicator_name, years):
    """Calculates the centralized and normalized excess kurtosis of distribution."""
    # Filter data
    data_filtered = data[(data['Country Name'].isin(countries)) & (data['Indicator'].isin(indicator_name))][['Country Name'] + years].fillna(0)

    # Prepare the DataFrame for kurtosis calculation
    data_prepared = data_filtered.melt(id_vars=['Country Name'], var_name='Year', value_name='Emissions')
    dist = data_prepared["Emissions"]

    # Calculate average and standard deviation for centralizing and normalizing
    aver = np.mean(dist)
    std = np.std(dist)

    # Calculate kurtosis
    value = np.sum(((dist - aver) / std)**4) / len(dist) - 3
    print("Kurtosis =", np.round(value, 6))
    return value

# Parameters for calculation
years_to_plot = ['2000']
selected_countries = ["Arab World"]
selected_indicator = ["Cereal yield (kg per hectare)"]

# Calculating skewness and kurtosis
skew_value = skew(data, selected_countries, selected_indicator, years_to_plot)
kurtosis_value = kurtosis(data, selected_countries, selected_indicator, years_to_plot)
