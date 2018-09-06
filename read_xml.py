import xml.etree.ElementTree as ET

tree = ET.parse('./dataset/Annotations/POD_0007.xml')

root = tree.getroot()

points = []
for child in root.findall('.//tableRegion'):
	coords = child.find('.//Coords')
	coordinates = coords.get('points')
	print(coordinates)
	coordinates = coordinates.split()
	print(coordinates)
	points = []
	for point in coordinates:
		point = point.split(',')
		points.append(point)
	# returning as dict: xmin, ymin, xmax, ymax
	print(points)
	new_points = {
		'xmin': points[0][0],
		'ymin': points[0][1],
		'xmax': points[3][0],
		'ymax': points[3][1]
	}


# print(coordinates)

print(new_points)

# print('Labels: [(x_min, y_min), :, :, (x_max, y_max)]')


