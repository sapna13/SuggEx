import csv

#returns a list of all the data instances, where each instance is a
# list of text and its label
def read_csv(inputFilePath):
    fileReader = open(inputFilePath, "r+", encoding = "utf-8", errors = "ignore")
    tweets_list = []

    for row in csv.reader(fileReader):
        list_element = [row[1],row[2]]
        tweets_list.append(list_element)

    return tweets_list

def write_csv(input_list, outputFilePath):
    csv_file = csv.writer(open(outputFilePath, "w+",newline=''))
    csv_file.writerow(['id', 'tweet', 'airline name'])

    firstline=False

    for element in input_list:
        if firstline:
            id = element[0]
            tweet = element[1]
            airline_name = element[2]
            csv_file.writerow([id, tweet, airline_name])
        firstline =True