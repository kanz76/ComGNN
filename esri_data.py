import numpy as np
import csv

class Esri_ascii_data:
    '''
        ESRI ASCII data
    '''
    def __init__(self, matrix, xllcorner, yllcorner, cellsize, nodata_value):
        self.matrix = matrix
        self.xllcorner = xllcorner
        self.yllcorner = yllcorner
        self.cellsize = cellsize
        self.nodata_value = nodata_value

def read_esri_ascii_raster(input_file_name, sep='\t'):
    '''
        ncols         1961
        nrows         1636
        xllcorner     239145.000000000000000
        yllcorner     3274515.000000000000000
        cellsize      30.000000000000000
        NODATA_value  -9999.000000
        
        https://en.wikipedia.org/wiki/Esri_grid
    '''
    with open(input_file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ',  skipinitialspace=True)
        line = next(reader)
        ncols = int(line[1])
        line = next(reader)
        nrows = int(line[1])
        line = next(reader)
        xllcorner = float(line[1])
        line = next(reader)
        yllcorner = float(line[1])
        line = next(reader)
        cellsize = float(line[1])
        line = next(reader)
        nodata_value = float(line[1])
                
        matrix = np.zeros((nrows, ncols))
        
        for i in range(nrows):
            line = next(reader)
            
            if sep == '\t':
                line = line[0].split(sep)
                
            matrix[i] = np.array(line[:-1])
    
    return Esri_ascii_data(matrix, xllcorner, yllcorner, cellsize, nodata_value)


def save_esri_data(esri_data, output_file_name, sep=' '):
    '''
        ncols         1961
        nrows         1636
        xllcorner     239145.000000000000000
        yllcorner     3274515.000000000000000
        cellsize      30.000000000000000
        NODATA_value  -9999.000000
    '''
    with open(output_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        
        writer.writerow(['ncols', esri_data.matrix.shape[1]])
        writer.writerow(['nrows', esri_data.matrix.shape[0]])
        writer.writerow(['xllcorner', esri_data.xllcorner])
        writer.writerow(['yllcorner', esri_data.yllcorner])
        writer.writerow(['cellsize', esri_data.cellsize])
        writer.writerow(['NODATA_value', esri_data.nodata_value])
        
        writer = csv.writer(csvfile, delimiter=sep)
        
        for i in range(esri_data.matrix.shape[0]):
            writer.writerow([str(val) for val in list(esri_data.matrix[i])]+[''])
    
