# kinds = ['formula','figure','table']
#
# from xml.etree import ElementTree
# class filedata:
#     def __init__(self):
#         self.pageLines = []
#         self.filename = ''
#         self.imgpath = ''
#     def setImagePath(self,imgpath):
#         self.imgpath = imgpath
#     def getTxtname(self):
#         return self.filename.split('\\')[-1]
#     def getImgname(self):
#         return self.imgpath.split('\\')[-1]
#     def readtxt(self,txtpath):
#         self.filename = txtpath
#         self.pageLines = []
#         root = ElementTree.parse(txtpath)
#         for name in ['figureRegion','formulaRegion','tableRegion']:
#             p = root.findall(name)
#             for line in p:
#                 coords = line.getchildren()
#                 line = coords[0].attrib['points']
#                 line = line.split(' ')
#                 kinds = name.split('Region')[0]
#                 region = rect(10000,0,10000,0)
#                 for xy in line:
#                     x,y = xy.split(',')
#                     region.update(int(x),int(y))
#                 self.pageLines.append(pageLine(region,kinds))
# class pageLine:
#     def __init__(self,rect,kind):
#         self.kind = kind
#         self.rect = rect
#         self.compList = []
#         self.prob = 0
#     def addComp(self,comp):
#         self.compList.append(comp)
#     # def contain(self,compRegion):
#         # if overlap(self.rect,compRegion.rect,compRegion.rect) > 0.5:
#         #     return True
#         # return False
#     def showComp(self):
#         for compRegion in self.compList:
#             print compRegion.tostr(),',',
#         print
#     # def sort(self):
#     #     self.compList.sort(lambda x,y: rectcmpLine(x.rect,y.rect))
#     def show(self):
#         print self.rect.tostr()+'\t'+self.kind
#
# class rect:
#     def __init__(self,l,r,u,d):
#         self.l = int(l); self.r = int(r)
#         self.u = int(u); self.d = int(d)
#     def update(self,x,y):
#         self.l = min(self.l,x)
#         self.r = max(self.r,x)
#         self.u = min(self.u,y)
#         self.d = max(self.d,y)
#
