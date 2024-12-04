from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType, StringType
import os
from process_image import processImage

sparkSession = SparkSession.builder.appName("Processor").getOrCreate()

inputFolders = {"locked": "./Locked", "unlocked": "./Unlocked"}
outputFolders = {"locked": "./CLocked", "unlocked": "./CUnlocked"}

imageData = []
for label, folder in inputFolders.items():
    for fileName in os.listdir(folder):
        filePath = os.path.join(folder, fileName)
        imageData.append((filePath, label))

imageDataFrame = sparkSession.createDataFrame(imageData, ["filePath", "label"])

def readAndProcessImage(filePath):
    image = cv2.imread(filePath)
    image = processImage(image)
    _, image = cv2.imencode(".jpg", image)

    return image.tobytes()

preprocessUdf = udf(preprocessImage, BinaryType())

processedDataFrame = imageDataFrame.withColumn("processedImage", preprocessUdf(imageDataFrame.filePath))

def saveImage(processedImage, outputFolder, originalFilePath):
    fileName = os.path.basename(originalFilePath)
    outputPath = os.path.join(outputFolder, fileName)

    with open(outputPath, "wb") as file:
        file.write(processedImage)

saveUdf = udf(lambda processedImage, label, filePath: saveImage(
    processedImage, outputFolders[label], filePath
), StringType())

processedDataFrame.select(
    saveUdf(processedDataFrame.processedImage, processedDataFrame.label, processedDataFrame.filePath)
).collect()

sparkSession.stop()
