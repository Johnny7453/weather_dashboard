

# Load necessary libraries
library(readxl)
library(dplyr)
library(lubridate)

################################################

cat("STARTING TO PREPROCESS THE DATA \n")

CREATE_PREDICTION_DATA = TRUE
prediction_data_since = "2024-07-15"

# Setting the working directory by input

cat("Please enter the working directory: ")
working_directory <- readline()

setwd(working_directory)

# The files should start like this
file_prefix <- "Oulu Oulunsalo Pellonp"

# Get the list of files in the directory
files <- list.files(pattern = paste0("^", file_prefix))

# Check if there are any matching files
if(length(files) == 0) {
  stop("No files found with the given prefix")
}

# If multiple files match, use the first one
excel_file <- files[1]

cat("Reading file:", excel_file, "\n")

# Read the Excel file
data <- read_excel(excel_file)

# Rename the columns
colnames(data) <- c("station", "year", "month", "day", "time", "Temperature",
                    "highest_temperature", "lowest_temperature", "Moisture", 
                    "Wind_speed", "Rain", "Air_pressure")

# Combine the time columns 
#and remove extra columns

data <- data %>%
  mutate(
    hour = as.integer(substr(time, 1, 2)),   # Extract hour
    minute = as.integer(substr(time, 4, 5)), # Extract minute
    Date = make_datetime(year, month, day, hour, minute)
  )%>% select(-station, -year, -month, -day, -time, -hour, -minute) 


if(CREATE_PREDICTION_DATA){
  data_prediction <- subset(data, Date >= as.Date(prediction_data_since))
  data <- subset(data, Date < as.Date(prediction_data_since)) 
  write.csv(data_prediction, "temperature_prediction_data.csv", row.names = FALSE)
}

# Save the modified data to a CSV file
write.csv(data, "temperature.csv", row.names = FALSE)


cat("Data has been processed and saved")


