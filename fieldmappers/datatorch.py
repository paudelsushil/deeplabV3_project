from datetime import datetime

from torch.utils.data import Dataset, DataLoader

from .utils import *
from .augmentation import *
from .tools import parallelize_df


class ImageData(Dataset):
    '''
    Dataset of image files (typically Planet) for pytorch architecture
    '''

    def __init__(self, data_path, log_dir, catalog, data_size, buffer, 
                 buffer_comp, usage, img_path_cols, 
                 norm_stats_type="local_per_tile",
                 label_path_col=None, label_group=[0, 1, 2, 3, 4], 
                 catalog_index=None, rotate=(-90, 90),
                 bright_shift=(4, 4), trans=None, parallel=False, 
                 downfactor = 32
                ):

        '''
        Params:
            data_path : str
                Directory storing files of variables and labels.
            log_dir : str 
                Directory to save the log file.
            catalog : Pandas.DataFrame 
                Pandas dataframe giving the list of data and their directories
            data_size : int 
                Size of chips that is not buffered, i.e., the size of labels
            buffer : int
                Distance to target chips' boundaries measured by 
                number of pixels when extracting images (variables), i.e., 
                variables size would be (dsize + buffer) x (dsize + buffer)
            buffer_comp : int 
                Buffer used when creating composite. In the case of Ghana, 
                it is 11.
            usage : str 
                Usage of the dataset : "train", "validate" or "predict"
            img_path_cols : list 
                Column names in the catalog referring to image paths
            label_path_col : str 
                Column name in the catalog referring to label paths
            label_group : list 
                Group indices of labels to load, where each group corresponds 
                to a specific level of label quality
            catalog_index : int or None 
                Row index in catalog to load data for prediction. Only need to 
                be specified when usage is "prediction"
            rotate : tuple or None 
                Range of degrees for rotation
            bright_shift : tuple or list
                Number of bands or channels on dataset for each brightness shift
            trans : list 
                Data augmentation methods: one or multiple elements from 
                ['vflip','hflip','dflip', 'rotate', 'resize']
            parallel : bool
                Whether to load in parallel (True) or not (False)
            downfactor : int
                The total downsampling factor in the network, used to 
                check that image chips are evenly divisible by that number
        Note:
            Provided transformation are:
                1) 'vflip', vertical flip
                2) 'hflip', horizontal flip
                3) 'dflip', diagonal flip
                4) 'rotate', rotation
                5) 'resize', rescale image fitted into the specified data size
                6) 'shift_brightness', shift brightness of images
            Any value out of the range would cause an error
        Note:
            Catalog for train and validate contrains at least columns for image
            path, label path and "usage".
            Catalog for prediction contains at least columns for image path, 
            "tile_col", and "tile_row", where the "tile_col" and "tile_row" is 
            the relative tile location for naming predictions in Learner
        '''

        self.buffer = buffer
        self.composite_buffer = buffer_comp
        self.data_size = data_size
        self.chip_size = self.data_size + self.buffer * 2

        self.usage = usage
        self.rotate = rotate
        self.bshift_subs = bright_shift
        self.trans = trans
        self.norm_stats_type = norm_stats_type

        self.data_path = data_path
        self.log_dir = log_dir
        self.img_cols = img_path_cols if isinstance(img_path_cols, list) \
            else [img_path_cols]
        self.label_col = label_path_col
        self.parallel = parallel
        self.downfactor = downfactor

        self.logger = setup_logger(
            self.log_dir, f"{self.usage}_dataset_report", use_date=False
        )
        start = datetime.now()
        msg = f'started dataset creation process at: {start}'
        progress_reporter(msg, verbose=False, logger=self.logger)

        if self.usage == "train":
            self.catalog = catalog.loc[
                (catalog['usage'] == self.usage) & 
                (catalog['label_group'].isin(label_group))
            ].copy()
            self.img, self.label = self.get_train_validate_data()

            end = datetime.now()
            msg = f'Completed dataset creation process at: {end}'
            progress_reporter(msg, verbose=False, logger=self.logger)
            s = '----------{} samples loaded in training dataset-----------'
            print(s.format(len(self.img)))

        elif self.usage == "validate":
            self.catalog = catalog.loc[
                (catalog['usage'] == self.usage) & 
                (catalog['label_group'].isin(label_group))
            ].copy()
            self.img, self.label = self.get_train_validate_data()

            end = datetime.now()
            msg = f'Completed dataset creation process at: {end}'
            progress_reporter(msg, verbose=False, logger=self.logger)
            s = '----------{} samples loaded in validation dataset-----------'
            print(s.format(len(self.img)))

        elif self.usage == "predict":
            self.catalog = catalog.iloc[catalog_index].copy()
            self.tile = (self.catalog['tile_col'], self.catalog['tile_row'])
            # self.year = self.catalog["dir_os"].split("_")[1].split("-")[0]
            self.year = self.catalog["year"]
            self.img, self.index, self.meta = self.get_predict_data()

        else:
            raise ValueError("Bad usage value")

    def get_train_validate_data(self):
        '''
        Get paris of image, label for train and validation
        Returns:
            tuple of list of images and label
        '''

        def load_label(row, data_path):
            buffer = self.buffer

            dir_label = row[self.label_col] \
                if row[self.label_col].startswith("s3") \
                else os.path.join(data_path, row[self.label_col])
            label = load_data(dir_label, is_label=True)
            label = np.pad(label, buffer, 'constant')
            msg = f'.. processing lbl sample: {os.path.basename(dir_label)}'\
                ' is complete.'
            progress_reporter(msg, verbose=False, logger=self.logger)

            return label

        def load_img(row, data_path):
            buffer = self.buffer

            dir_label = row['dir_label'] if row['dir_label'].startswith("s3")\
                else os.path.join(data_path, row['dir_label'])
            dir_imgs = [row[m] if row[m].startswith("s3") 
                        else os.path.join(data_path, row[m]) 
                        for m in self.img_cols]
            window = get_buffered_window(dir_imgs[0], dir_label, buffer)
            img = get_stacked_img(dir_imgs, self.usage, self.norm_stats_type, 
                                  window=window)

            msg = f'.. processing img sample: {os.path.basename(dir_imgs[0])}'\
                ' is complete.'
            progress_reporter(msg, verbose=False, logger=self.logger)

            return img

        if self.parallel:
            global list_data  # Local function not applicable in parallelism
            def list_data(catalog, data_path):
                catalog["img"] = catalog.apply(
                    lambda row: load_img(row, data_path), axis=1
                )
                catalog["label"] = catalog.apply(
                    lambda row: load_label(row, data_path), axis=1
                )
                return catalog.filter(items=['label', 'img'])
    
            catalog = parallelize_df(self.catalog, list_data, 
                                     data_path=self.data_path)
            
            img_ls = catalog['img'].tolist()
            label_ls = catalog['label'].tolist()
        
        else:
            self.catalog["img"] = self.catalog.apply(
                lambda row: load_img(row, data_path=self.data_path), axis=1
            )
            self.catalog["label"] = self.catalog.apply(
                lambda row: load_label(row, data_path=self.data_path), axis=1
            )
            img_ls = self.catalog['img'].tolist()
            label_ls = self.catalog['label'].tolist()
        
        return img_ls, label_ls
        
        """
        global list_data  # Local function not applicable in parallelism

        def list_data(catalog, data_path):
            catalog["img"] = catalog.apply(lambda row: 
            load_img(row, data_path), axis=1)
            catalog["label"] = catalog.apply(lambda row: 
            load_label(row, data_path), axis=1)

            return catalog.filter(items=['label', 'img'])

        catalog = parallelize_df(self.catalog, list_data, 
        data_path=self.data_path)

        img_ls = catalog['img'].tolist()
        label_ls = catalog['label'].tolist()

        return img_ls, label_ls
        """

    def get_predict_data(self):
        '''
        Get data for prediction
        Returns:
            list of cropped chips
            list of index representing location of each chip in tile
            dictionary of metadata of score map reconstructed from chips
        '''

        dir_imgs = [self.catalog[m] if self.catalog[m].startswith("s3") \
                    else os.path.join(self.data_path, self.catalog[m]) \
                    for m in self.img_cols]
        # entire composite image in (H, W, C)
        img = get_stacked_img(dir_imgs, self.usage, self.norm_stats_type)  
        buffer_diff = self.buffer - self.composite_buffer
        h, w, c = img.shape

        if buffer_diff > 0:
            canvas = np.zeros((h + buffer_diff * 2, w + buffer_diff * 2, c))

            for i in range(c):
                canvas[:, :, i] = np.pad(img[:, :, i], buffer_diff, 
                                         mode='reflect')
            img = canvas

        else:
            buffer_diff = abs(buffer_diff)
            img = img[buffer_diff:h - buffer_diff, 
                      buffer_diff:w - buffer_diff, :]

        # meta of composite buffer removed
        meta = get_meta_from_bounds(dir_imgs[0], self.composite_buffer)  
        img_ls, index_ls = get_chips(img, self.chip_size, self.buffer)
        
        if img_ls[0].shape[0] % self.downfactor != 0:
            assert f"Chip is not evenly divisible by {self.downfactor}"

        return img_ls, index_ls, meta

    def __getitem__(self, index):
        """
        Support dataset indexing and apply transformation
        Args:
            index -- Index of each small chips in the dataset
        Returns:
            tuple
        """

        if self.usage in ["train", "validate"]:
            img = self.img[index]
            label = self.label[index]

            if self.usage == "train":
                mask = np.pad(np.ones((self.data_size, self.data_size)), 
                              self.buffer, 'constant')
                trans = self.trans
                # trans = None
                rotate = self.rotate

                if trans:

                    # 0.5 possibility to flip
                    trans_flip_ls = [m for m in trans if 'flip' in m]
                    if random.randint(0, 1) and len(trans_flip_ls) > 1:
                        trans_flip = random.sample(trans_flip_ls, 1)
                        img, label, mask = flip(img, label, mask, 
                                                trans_flip[0])

                    # 0.5 possibility to resize
                    if random.randint(0, 1) and 'resize' in trans:
                        img, label, mask = reScale(
                            img, label.astype(np.uint8), mask.astype(np.uint8),
                            randResizeCrop=True, diff=True, cenLocate=False
                        )

                    # 0.5 possibility to rotate
                    if random.randint(0, 1) and 'rotate' in trans:
                        img, label, mask = centerRotate(img, label, mask, 
                                                        rotate)

                    # 0.5 possibility to shift brightness
                    if random.randint(0, 1) and 'shift_brightness' in trans:
                        img = shiftBrightness(
                            img, gammaRange=(0.2, 2), 
                            shiftSubset=self.bshift_subs, 
                            patchShift=True
                        )

                # numpy to torch
                label = torch.from_numpy(label).long()
                mask = torch.from_numpy(mask).long()
                img = torch.from_numpy(img.transpose((2, 0, 1))).float()

                # display(img[:, self.buffer:-self.buffer, 
                # self.buffer:-self.buffer], label[self.buffer:-
                # self.buffer,self.buffer:-self.buffer], 
                # mask[self.buffer:-self.buffer,self.buffer:-self.buffer])
                # display(img, label, mask)

                return img, label, mask

            else:
                # numpy to torch
                label = torch.from_numpy(label).long()
                img = torch.from_numpy(img.transpose((2, 0, 1))).float()

                return img, label

        else:
            img = self.img[index]
            index = self.index[index]
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()

            return img, index

    def __len__(self):
        '''
        Get size of the dataset
        '''

        return len(self.img)


def load_dataset(params, usage, catalog_row=None):
    if usage not in ["train", "validate", "predict"]:
        raise ValueError("Bad usage, should be 'train', 'validate', 'predict'")

    dir_data = params['dir_data']
    log_dir = params['log_dir']
    # catalog = pd.read_csv(os.path.join(dir_data, params['catalog']))
    catalog = pd.read_csv(params['catalog'])

    buffer = params['buffer']
    composite_buffer = params['composite_buffer']
    patch_size = params['patch_size']
    img_path_cols = params['img_path_cols']

    # train
    if usage == "train":
        label_path_col = params['label_path_col']
        train_batch = params['train_batch']
        rotate = params['rotation_degrees']
        transformation = params['transformation']

        dataset = ImageData(
            dir_data, log_dir, catalog, patch_size, buffer, composite_buffer, 
            "train", img_path_cols=img_path_cols, 
            label_path_col=label_path_col, label_group=params["train_group"], 
            rotate=rotate, bright_shift=params['brightness_shift_subsets'], 
            trans=transformation
        )
        data_loader = DataLoader(dataset, batch_size=train_batch, 
                                 shuffle=True)

        return data_loader

    # validate
    elif usage == "validate":
        label_path_col = params['label_path_col']
        val_batch = params['validate_batch']
        dataset = ImageData(
            dir_data, log_dir, catalog, patch_size, buffer, composite_buffer, 
            "validate", img_path_cols=img_path_cols, 
            label_path_col=label_path_col, label_group=params["validate_group"]
        )
        data_loader = DataLoader(dataset, batch_size=val_batch, shuffle=False)

        return data_loader

    # prediction
    else:

        pred_batch = params['pred_batch']

        def load_single_tile(catalog_ind=catalog_row):
            dataset = ImageData(
                dir_data, log_dir=log_dir, catalog=catalog, 
                data_size=patch_size, buffer=buffer, 
                buffer_comp=composite_buffer, usage="predict", 
                catalog_index=catalog_ind, 
                img_path_cols=img_path_cols
            )
            data_loader = DataLoader(dataset, batch_size=pred_batch, 
                                     shuffle=False)
            meta = dataset.meta
            tile = dataset.tile
            year = dataset.year
            return data_loader, meta, tile, year

        def catalog_query(catalog, catalog_row, direction):
            tile_col = catalog.iloc[catalog_row].tile_col
            tile_row = catalog.iloc[catalog_row].tile_row
            q = catalog.query(
                f"tile_col=={tile_col} & tile_row=={tile_row-1}"
            ).iloc[0].name 

            if direction == "top":
                colrow = f"{tile_col}_{tile_row-1}"
            elif direction == "left":
                colrow = f"{tile_col-1}_{tile_row}"
            elif direction == "right":
                colrow = f"{tile_col+1}_{tile_row}"
            else: 
                colrow = f"{tile_col}_{tile_row+1}"
            
            return q if colrow in list(catalog.tile_col_row) else None

        # average neighbor
        if params['average_neighbors'] == True:
            catalog["tile_col_row"] = catalog.apply(
                lambda x: "{}_{}".format(x['tile_col'], x['tile_row']), axis=1
            )
            # tile_col = catalog.iloc[catalog_row].tile_col
            # tile_row = catalog.iloc[catalog_row].tile_row
            row_dict = {
                "center": catalog_row,
                "top": catalog_query(catalog, catalog_row, "top"),
                "left": catalog_query(catalog, catalog_row, "left"),
                "right": catalog_query(catalog, catalog_row, "right"),
                "bottom": catalog.query(catalog, catalog_row, "bottom"),
            }
            dataset_dict = {
                k: load_single_tile(catalog_ind=row_dict[k]) \
                    if row_dict[k] is not None else None
                    for k in row_dict.keys()
                }
            return dataset_dict
        # direct crop edge pixels
        else:
            return load_single_tile()
