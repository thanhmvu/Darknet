import glob

train_path = "./src/";
result_path = "./results/";

def css():
	f = open("style.css","w")
	contents = """
	<style >
	table {
	    font-family: arial, sans-serif;
	    border-collapse: collapse;
	    width: 100%;
	}

	td, th {
		white-space: nowrap;
	    border: 1px solid #dddddd;
	    text-align: left;
	    padding: 8px;
	}

	tr:nth-child(even) {
	    background-color: #dddddd;
	}
	</style>
	"""

	f.write(contents)


def html(contents):
	f = open("results.html","w")
	
	display = """
	<!DOCTYPE html>
	<html>
	<head>
		<link rel="stylesheet" href="style.css">
	</head>
	<body>
	""" + contents + "</body>\n</html>\n"

	f.write(display)

def htmlImg(img, borderColor):
	return "<img src=\""+ img + "\" alt=\"Class " + img + "\" style=\"height:200px; border:3px solid "+ borderColor +";\">"

def imgIdx(img):
	return img.split("/")[-1].split(".")[0]

def display(resultDir,trainDir):
	resultTable = """
	<table>
	  <tr>
	    <th>Class Index</th>
	    <th>Correct Images</th>
	    <th>Training Image</th>
	    <th>Test Images</th>
	  </tr>
	"""

	totalCorrectImgs = 0
	totalIncorrectImgs = 0

	for folder in glob.glob(resultDir + "*"):
		classIdx = folder.split("/")[-1]
		trainImg = trainDir + classIdx + ".jpg";
		correctImgs = glob.glob(folder + "/1/*");
		incorrectImgs = glob.glob(folder + "/0/*");
		resultTable += """
		<tr>
			<th>"""+ classIdx +"""</th>
			<th>"""+ `len(correctImgs)` +"""</th>
			<th>"""+ htmlImg(trainImg,"transparent") +"""</th>
		"""
		for img in correctImgs:
			resultTable += "<th>" +htmlImg(img,"blue") +"<br>"+ imgIdx(img) +"</th>"
		for img in incorrectImgs:
			resultTable += "<th>" +htmlImg(img,"red") +"<br>"+ imgIdx(img) +"</th>"
		resultTable += "</tr>"
		
		totalCorrectImgs += len(correctImgs)
		totalIncorrectImgs += len(incorrectImgs)

	accuracy = totalCorrectImgs*1.0/(totalCorrectImgs +	totalIncorrectImgs)
	resultTable += "</table>\n"
	contents = "Total accuracy: " +`accuracy*100` +"%<br><br>" +resultTable
	
	css()
	html(contents)



# ============================== Main ==============================
display(result_path,train_path)


