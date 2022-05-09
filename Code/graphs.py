###### os- file management
import os
###### - Pandas - dataframe
import pandas as pd

###### - Pandas - dataframe
import numpy as np

###### - Mathplot
import matplotlib.pyplot as plt

###### - tFidfVect for word processing
from sklearn.feature_extraction.text import TfidfVectorizer

# Current working directory
directory = os.getcwd()

# Define graphs colors
colorList = []
colorList.append("aquamarine")
colorList.append("c")
colorList.append("lightseagreen")
colorList.append("cornflowerblue")
colorList.append("aquamarine")
colorList.append("c")
colorList.append("lightseagreen")
# Load data
data = pd.read_csv(directory[:-4] + "results.csv")


# Describe data
# print(data.to_string())
# print(data.info())
# print(data.describe())


# Scatter plot
def matPlScat(pPlt, pData, pMaxR, pMaxC, pFigure, pColorList, pTitle, pSaveRute, pDataTitle):
    row = 0
    col = 0
    n = 1
    pPlt.figure(pFigure)
    fig, ax = pPlt.subplots(pMaxR, pMaxC)
    fig.suptitle(pTitle,fontsize=24)
    for (columnName, columnData) in pData.iteritems():
        if (columnName != pDataTitle):
            ax[row, col].scatter(columnData, pData[pDataTitle], color=pColorList[n - 1])
            ax[row, col].set_title(str(n) + "." + columnName)
            if (col < pMaxC - 1):
                col += 1
            else:
                col = 0
                row += 1
            n += 1
    # Eliminate empty subplots
    if (pData.shape[1] < pMaxR * pMaxC):
        col = pMaxC - 1
        row = pMaxR - 1
        while (pData.shape[1] < row * pMaxC + col):
            ax[row, col].set_axis_off()
            if (col == 0):
                col = pMaxC - 1
                row += -1
            else:
                col += -1
    # pPlt.show()
    fig.set_size_inches(30 / 2.54, 30 / 2.54)
    savefig = pPlt.savefig(pSaveRute)


# Histogram plot
def matPlHisto(pPlt, pData, pMaxR, pMaxC, pFigure, pColorList, pTitle, pSaveRute):
    row = 0
    col = 0
    n = 1
    pPlt.figure(pFigure)
    fig, ax = pPlt.subplots(pMaxR, pMaxC)
    fig.suptitle(pTitle,fontsize=24)
    for (columnName, columnData) in pData.iteritems():
        ax[row, col].hist(columnData, bins=8, linewidth=0.5, edgecolor="white", color=pColorList[n - 1])
        ax[row, col].set_title(str(n) + "." + columnName)
        if (col < pMaxC - 1):
            col += 1
        else:
            col = 0
            row += 1
        n += 1
    # Eliminate empty subplots
    if (pData.shape[1] < pMaxR * pMaxC):
        col = pMaxC - 1
        row = pMaxR - 1
        while (pData.shape[1] < row * pMaxC + col + 1):
            ax[row, col].set_axis_off()
            if (col == 0):
                col = pMaxC - 1
                row += -1
            else:
                col += -1
    # pPlt.show()
    fig.set_size_inches(30 / 2.54, 30 / 2.54)
    pPlt.savefig(pSaveRute)

def simplePieChart(pPlt,x,pLabels,pTitle, pSaveRute,pAutopct,pColors,pExplode,pLegTitle):
    pPlt.figure(1)
    pPlt.rcParams['font.size'] = 20
    pPlt.title(pTitle, fontsize=24)
    wedges, texts, autotexts=pPlt.pie(x,autopct=pAutopct,explode=pExplode, colors=pColors,shadow=True,wedgeprops = {"edgecolor" : "black",
                      'linewidth': 2,
                      'antialiased': True})
    pPlt.legend(wedges,pLabels,
              title=pLegTitle,
              bbox_to_anchor=(1.1, 0.05))
    pPlt.savefig(pSaveRute)
#From https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
def scatter_hist(x, y, ax, ax_histx, ax_histy,pColor):
    # no labels
    ax_histy.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histx.yaxis.set_tick_params(labelleft=False)
    # the scatter plot:
    ax.scatter(x, y,color=pColor)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax_histx.hist(x,edgecolor='white',color=pColor)
    ax_histy.hist(y, orientation='horizontal',edgecolor='white',color=pColor)
def printScatterHist(x,y,pPlt,pTitle,pSavePath,pXTitle,pYTitle,pColor):
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = pPlt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function

    scatter_hist(x, y, ax, ax_histx, ax_histy, pColor)
    ttl = ax_histx.set_title(pTitle, fontsize=24, )
    ttl.get_horizontalalignment = 'right'
    ax.yaxis.set_label_text(pYTitle)
    ax.xaxis.set_label_text(pXTitle)

    pPlt.savefig(pSavePath)


def printHeatMap(pData, pXCol, pYCol, pZCol, pNumXDiv, pNumYDiv, pPlt, pTitle, pXTitle, pYTitle, pSavePath, pXFormat,
                 pYFormat):
    maxX, minX = max(pData[pXCol]), min(pData[pXCol])
    maxY, minY = max(pData[pYCol]), min(pData[pYCol])
    divX, divY = (maxX - minX) / pNumXDiv, (maxY - minY) / pNumYDiv
    pExtent = [-1, 1, -1, 1]
    values = np.zeros((pNumYDiv,pNumXDiv))
    for ix in range(pNumXDiv):
        for iy in range(pNumYDiv):
            pDataTemp = pData[(pData[pXCol] > minX + ix * divX) & (pData[pXCol] <= minX + (ix + 1) * divX) & (
                        pData[pYCol] > minY + iy * divY) & (pData[pYCol] <= minY + (iy + 1) * divY)]
            values[iy][ix] = pDataTemp[pZCol].mean()
    # Get figure and ax objects
    fig, ax = pPlt.subplots(1, 1)
    # Adds title
    plt.rcParams.update({'font.size': 8})
    pPlt.title(pTitle, fontsize=9)
    # Creates plot
    img = ax.imshow(values)
    # Creates x labels
    x_label_list = []
    x_tick_list = []
    for ix in range(pNumXDiv):
        x_label_list.append(
            "(" + pXFormat.format(minX + ix * divX) + "," + pXFormat.format(minX + (ix + 1) * divX) + "]")
        x_tick_list.append(ix)

    # Creates y labels
    y_label_list = []
    y_tick_list = []
    for iy in range(pNumYDiv):
        y_label_list.append(
            "(" + pYFormat.format(minY + iy * divY) + "," + pYFormat.format(minY + (iy + 1) * divY) + "]")
        y_tick_list.append(iy)

    # Set labels
    ax.set_xticks(x_tick_list)
    ax.set_xticklabels(x_label_list, fontsize=8)
    ax.set_yticks(y_tick_list)
    ax.set_yticklabels(y_label_list, fontsize=8)
    ax.set_xlabel(pXTitle, fontsize=8)
    ax.set_ylabel(pYTitle, fontsize=8)
    fig.colorbar(img)
    pPlt.savefig(pSavePath)

def cleanFigures(pPlt):
    pPlt.close('all')

#### Main ####
# 1: normal histograms
maxR, maxC = 2, 4
matPlHisto(plt, data, maxR, maxC, 0, colorList, "Histograms of each variable", directory[:-4] + "graphs\O_histograms.png")
plt.clf()
# 2: histograms w/ test_accuracy under threshold
threshold=90
matPlHisto(plt, data[data["test_accuracy"] < threshold], maxR, maxC, 0, colorList, "Histograms w/test_accuracy under "+str(threshold)+"%",
           directory[:-4] + "graphs\less"+str(threshold)+"_histograms.png")
plt.clf()
# 3: histograms w/ test_accuracy over 90%
matPlHisto(plt, data[data["test_accuracy"] >= threshold], maxR, maxC, 0, colorList, "Histograms w/test_accuracy under "+str(threshold)+"%",
           directory[:-4] + "graphs\more"+str(threshold)+"_histograms.png")
plt.clf()
# 4: Piechart proportion of under and over 90%
x=[data[data["test_accuracy"] < threshold].shape[0],data[data["test_accuracy"] >= threshold].shape[0]]
labels=["Under", "Equal or over"]
plt.clf()
explode=(0.5,0)
#plt.rcParams.update({'font.size': 20})
simplePieChart(plt,x,labels,"Distribution of models with "+str(threshold)+"% or more accuracy",
               directory[:-4] + "graphs\PieChart"+str(threshold)+".png",'%0.00f%%',["darkorange","dodgerblue"],explode
               ,"Models with "+str(threshold)+"% accuracy")
plt.clf()

#5: Piechart proportion of under and over 93%
x=[data[data["test_accuracy"] < 93].shape[0],data[data["test_accuracy"] >= 93].shape[0]]
labels=["Under", "Equal or over"]
plt.clf()
explode=(0,0.25)
simplePieChart(plt,x,labels,"Distribution of models with 93% or more accuracy", directory[:-4] + "graphs\PieChart93.png"
               ,'%0.00f%%',["darkorange","dodgerblue"],explode,"Models with 93% accuracy")


#6: scatter histogram of accuracy rate over 90% vs each other column
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
xData=data[data["test_accuracy"] >= threshold]
x=xData["test_accuracy"]
xTitle="Test accuracy"
colorList2=[]
colorList2.append("limegreen")
colorList2.append("red")
colorList2.append("blue")
colorList2.append("m")
colorList2.append("orange")
colorList2.append("steelblue")
n=0
for (columnName, columnData) in xData.iteritems():
    if(columnName!="test_accuracy"):
        y=columnData
        yTitle = columnName
        savePath=directory[:-4] + "graphs\ScatHistAccVs"+ columnName +".png"
        title="Accuracy vs " + columnName
        color=colorList2[n]
        n+=1
        plt.clf()
        printScatterHist(x,y,plt,title,savePath,xTitle,yTitle,color)

#7: scatter histogram of accuracy rate under threshold% vs each other column
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
xData=data[data["test_accuracy"] < threshold]
x=xData["test_accuracy"]
xTitle="Test accuracy"
n=0
for (columnName, columnData) in xData.iteritems():
    if(columnName!="test_accuracy"):
        y=columnData
        yTitle = columnName
        savePath=directory[:-4] + "graphs\ScatHistAccVs"+ columnName +"Under"+str(threshold)+".png"
        title="Accuracy vs " + columnName
        color=colorList2[n]
        n+=1
        plt.clf()
        printScatterHist(x,y,plt,title,savePath,xTitle,yTitle,color)

# 8: Heat map of two variables vs accuracy rate and training_time over threshold%
xData = data[data["test_accuracy"] >= threshold]
comparision=np.zeros((xData.shape[1],xData.shape[1]))
normFor = "{:.0f}"
lrFor = "{:.3f}"
for (xCol, columnDataX) in xData.iteritems():
    if (xCol != "test_accuracy" and xCol != "training_time" and xCol != "testing_time"):
        for (yCol, columnDataY) in xData.iteritems():
            if (yCol != "test_accuracy" and yCol != "training_time" and yCol != "testing_time" and yCol!=xCol):
                #Checks if the two columns haven't been compared yet
                if(comparision[xData.columns.get_loc(xCol)][xData.columns.get_loc(yCol)]==0 and comparision[xData.columns.get_loc(yCol)][xData.columns.get_loc(xCol)]==0):
                    # if the two columns haven't been compared yet it update the matrix and continue with the graphs
                    comparision[xData.columns.get_loc(xCol)][xData.columns.get_loc(yCol)]=1
                    comparision[xData.columns.get_loc(yCol)][xData.columns.get_loc(xCol)]=1
                    # Creates test accuracy graph
                    zCol = "test_accuracy"
                    title = zCol + " over " + str(threshold) + "% with " + xCol + " and " + yCol
                    savePath = directory[:-4] + "graphs\HeatMap_" + zCol + " with_ " + xCol + "vs_" + yCol + "Over" + str(
                        threshold) + ".png"
                    cleanFigures(plt)
                    # Assigns the correct format to the x axis labels
                    if (xCol != "learning_rate"):
                        xFormat = normFor
                    else:
                        xFormat = lrFor

                    # Assigns the correct format to the y axis labels
                    if (yCol != "learning_rate"):
                        yFormat = normFor
                    else:
                        yFormat = lrFor
                    printHeatMap(xData, xCol, yCol, zCol, 6, 6, plt, title, xCol, yCol, savePath, xFormat, yFormat)

                    # Creates training time graph
                    zCol = "training_time"
                    title = zCol + " over " + str(threshold) + "% with " + xCol + " and " + yCol
                    savePath = directory[:-4] + "graphs\HeatMap_" + zCol + " with " + xCol + "vs" + yCol + "Under" + str(
                        threshold) + ".png"
                    cleanFigures(plt)
                    # Assigns the correct format to the x axis labels
                    if (xCol != "learning_rate"):
                        xFormat = normFor
                    else:
                        xFormat = lrFor

                    # Assigns the correct format to the y axis labels
                    if (yCol != "learning_rate"):
                        yFormat = normFor
                    else:
                        yFormat = lrFor
                    printHeatMap(xData, xCol, yCol, zCol, 8, 8, plt, title, xCol, yCol, savePath, xFormat, yFormat)
# 9. histogram for news in words to vector:
#Import data
path = directory[:-4]+"news.csv"
df = pd.read_csv(path)
dfT=df['text']
#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)
wordV=tfidf_vectorizer.fit_transform(dfT)
feature_names = tfidf_vectorizer.get_feature_names()
doc = 0
feature_index = wordV[doc,:].nonzero()[1]
numWords=10
actW=0
words=[]
scores=[]
tfidf_scores = zip(feature_index, [wordV[doc, x] for x in feature_index])
for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
    if(actW>=numWords):
        break
    else:
        actW+=1
    words.append(w)
    scores.append(s)
cleanFigures(plt)
title="10 First non-zero frequency words"
plt.bar(words,scores,color="darkblue")
plt.title(title, fontsize=20)
plt.ylabel("Tf-idf-weight (frequency)")
savePath=directory[:-4] + "graphs\BarWord.png"
plt.tight_layout()
plt.savefig(savePath)
cleanFigures(plt)

# 10: histograms w/ test_accuracy over threshold
threshold=93
matPlHisto(plt, data[data["test_accuracy"] >= threshold], maxR, maxC, 0, colorList, "Histograms w/test_accuracy over "+str(threshold)+"%",
           directory[:-4] + "graphs\over"+str(threshold)+"_histograms.png")