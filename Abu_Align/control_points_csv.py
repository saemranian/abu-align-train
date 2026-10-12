"""Explicit RealityScan measurement/XYZ CSV import; no projection guessing."""
import copy
import csv
import io
import math
import re
import shlex
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
from control_points_model import validate_controls

ALIASES = {'image': {'image','imagepath','imagename','filename','photo','shotid'},
           'name': {'point','pointname','name','controlpoint','id'}, 'x': {'x','east','easting','xpx'},
           'y': {'y','north','northing','ypx'}, 'z': {'z','alt','altitude','height','elevation'},
           'sigma': {'sigma','accuracy','sigmapx','sigmam','uncertainty'}}


def read_table(path):
    path = Path(path)
    if path.stat().st_size > 25*1024**2: raise ValueError('CSV exceeds the 25 MiB import budget; split it into smaller files.')
    raw = path.read_bytes()
    encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    try: text = raw.decode(encoding)
    except UnicodeDecodeError: text = raw.decode('cp1252')
    try: dialect = csv.Sniffer().sniff(text[:8192], delimiters=',;\t'); delimiter = dialect.delimiter
    except csv.Error: delimiter = 'space'
    rows = []
    if delimiter == 'space':
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith('#'): continue
            lexer = shlex.shlex(line, posix=True); lexer.whitespace_split = True; lexer.escape = ''; lexer.commenters = ''
            rows.append(list(lexer))
    else:
        rows = [[v.strip() for v in row] for row in csv.reader(io.StringIO(text), delimiter=delimiter,skipinitialspace=True) if row and any(v.strip() for v in row)]
    if not rows: raise ValueError('The CSV contains no rows.')
    if len(rows) > 100000: raise ValueError('Import at most 100000 rows at a time.')
    normalize = lambda value: re.sub('[^a-z0-9]', '', value.casefold())
    known = set().union(*ALIASES.values())
    header = sum(normalize(v) in known for v in rows[0]) >= 2
    headers = rows.pop(0) if header else [f'Column {i+1}' for i in range(max(map(len, rows)))]
    rows = [r for r in rows if not (r[0].startswith('#') and len(r)==1)]
    mapping = {}
    if header:
        for key, aliases in ALIASES.items():
            mapping[key] = next((i for i, value in enumerate(headers) if normalize(value) in aliases), None)
    measurement = 'image' in mapping and mapping.get('image') is not None
    if not header:
        image_like = Path(rows[0][0].replace('\\','/')).suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.webp','.bmp'}
        measurement = image_like
        mapping.update(name=1 if image_like else 0, image=0 if image_like else None,
                       x=2 if image_like else 1, y=3 if image_like else 2, z=None if image_like else 3, sigma=None)
    source_shape=None
    if measurement and rows and mapping.get('image') is not None:
        value=rows[0][mapping['image']].strip()
        source=Path(value)
        if not value.startswith(('\\\\','//')) and source.is_file():
            try:
                with Image.open(source) as image:source_shape=list(image.size)
            except OSError:pass
    return {'path': str(path), 'headers': headers, 'rows': rows, 'mapping': mapping,'source_shape':source_shape,
            'kind': 'measurements' if measurement else 'xyz', 'delimiter': delimiter, 'encoding': encoding}


def import_table(table, existing, image_sizes, kind='measurements', mapping=None, replace=False,
                 names_only=False, default_type='gcp', unit_scale=1., default_sigma=1., skip_unmatched=False):
    mapping = mapping or table['mapping']; data = copy.deepcopy(existing)
    points = {p['id']: p for p in data['points']}
    images = {}
    for name in image_sizes: images.setdefault(name.casefold(), []).append(name)
    stats = {'rows': len(table['rows']), 'new_points': 0, 'added_measurements': 0,
             'replaced_measurements': 0, 'kept_measurements': 0, 'skipped_images': 0, 'names_only': names_only}
    dimensions = {}
    seen={}
    def value(row, key, default=None):
        column = mapping.get(key)
        if column is None: return default
        if column >= len(row): raise ValueError(f'Missing {key} column in a CSV row.')
        return row[column].strip()
    def number(row, key, default=None):
        result = float(value(row, key, default))
        if not math.isfinite(result): raise ValueError(f'{key} must be finite.')
        return result
    for index, row in enumerate(table['rows'], 1):
        try:
            name = value(row, 'name')
            if not name: raise ValueError('Point name is empty.')
            shot = None; pixel = None; position = None
            if not names_only and kind == 'measurements':
                source = value(row, 'image')
                if not source: raise ValueError('Map the image filename column.')
                basename = source.replace('\\','/').rsplit('/',1)[-1]
                matches = images.get(basename.casefold(), [])
                if len(matches) != 1:
                    if skip_unmatched: stats['skipped_images'] += 1; continue
                    raise ValueError(f'Image {basename} is not uniquely matched. Open matching panoramas, or choose Point names only.')
                shot = matches[0]; width, height = image_sizes[shot]
                path = Path(source)
                if not source.startswith(('\\\\','//')) and path.is_file():
                    if source not in dimensions:
                        with Image.open(path) as image: dimensions[source] = image.size
                    sw, sh = dimensions[source]
                    if sw != 2*sh: raise ValueError('CSV references perspective/cropped images. Import point names and re-mark panoramas; pixel transfer needs calibrated projection mapping.')
                    if (sw,sh) != (width,height): raise ValueError('Export-image resolution differs from the selected originals. Rescale measurements explicitly before importing.')
                pixel = [number(row,'x'), number(row,'y')]
                if not 0 <= pixel[0] < width or not 0 <= pixel[1] < height: raise ValueError(f'{basename}: pixel outside {width}×{height}. Check export resolution and projection.')
            elif not names_only and kind == 'xyz':
                if any('lat' in h.casefold() or 'lon' in h.casefold() for h in table['headers']):
                    raise ValueError('Geographic coordinates need conversion to one local Cartesian frame before import.')
                position = [number(row,key)*unit_scale for key in ['x','y','z']]
            if not names_only:
                identity=(name,shot if kind=='measurements' else 'XYZ')
                coordinates=pixel if kind=='measurements' else position
                if identity in seen and seen[identity]!=coordinates:raise ValueError('Conflicting duplicate point/image or XYZ rows; resolve them in the CSV before import.')
                seen[identity]=coordinates
            if name not in points:
                point = {'id': name, 'type': 'tie', 'enabled': True, 'observations': []}
                data['points'].append(point); points[name] = point; stats['new_points'] += 1
            point = points[name]
            if names_only:
                point.setdefault('import_reference', {'file': Path(table['path']).name, 'note': 'Names imported; mark this feature on the original panoramas.'})
            elif kind == 'measurements':
                prior = next((o for o in point['observations'] if o['shot_id'] == shot), None)
                sigma = number(row, 'sigma', default_sigma)
                if sigma <= 0: raise ValueError('Image uncertainty must be positive.')
                if prior and not replace:
                    stats['kept_measurements'] += 1; continue
                if prior: stats['replaced_measurements'] += 1; point['observations'].remove(prior)
                else: stats['added_measurements'] += 1
                point['observations'].append({'shot_id':shot, 'pixel':pixel, 'sigma_px':sigma})
            else:
                sigma = number(row, 'sigma', default_sigma)*unit_scale
                point.update(type=default_type, position=position, sigma_m=sigma)
        except (ValueError, TypeError) as error: raise ValueError(f'CSV data row {index}: {error}') from error
    if not stats['new_points'] and not stats['added_measurements'] and not stats['replaced_measurements'] and kind != 'xyz':
        raise ValueError('Nothing new to import. Check image matching or choose replacement mode.')
    return validate_controls(data), stats


class CSVImportDialog:
    def __init__(self, editor, path):
        self.editor, self.table = editor, read_table(path)
        self.window = tk.Toplevel(editor.window); self.window.title('Import RealityScan control points'); self.window.geometry('920x710'); self.window.minsize(760,620)
        self.window.configure(bg='#1a2024'); self.candidate = None
        body = tk.Frame(self.window,bg='#1a2024',padx=16,pady=12); body.pack(fill='both',expand=True)
        editor.label(body, f'{Path(path).name} · {len(self.table["rows"])} rows · {self.table["delimiter"]} separated').pack(fill='x')
        shape=self.table.get('source_shape');incompatible=shape and shape[0]!=2*shape[1]
        self.kind = tk.StringVar(master=self.window,value='Point names only' if incompatible else 'Image measurements' if self.table['kind']=='measurements' else 'Local XYZ coordinates')
        if incompatible:editor.label(body,f'Referenced images: {shape[0]} × {shape[1]} · not panoramas. Names-only import selected; re-mark on your original panoramas.').pack(fill='x',pady=8)
        choice = editor.field(body,'Import content',self.kind,'Measurement CSV contains image/name/X/Y pixel observations. XYZ contains local reference positions. Names only reuses point identities without transferring incompatible pixels.', ['Image measurements','Local XYZ coordinates','Point names only'])
        choice.bind('<<ComboboxSelected>>',lambda e:self.change_kind())
        maps = tk.Frame(body,bg='#1a2024'); maps.pack(fill='x',pady=8); self.columns = {}
        labels = ['— Not used —']+[f'{i+1}: {h}' for i,h in enumerate(self.table['headers'])]
        for i,key in enumerate(['name','image','x','y','z','sigma']):
            cell=tk.Frame(maps,bg='#1a2024');cell.grid(row=i//3,column=i%3,sticky='ew',padx=5);maps.columnconfigure(i%3,weight=1)
            column=self.table['mapping'].get(key);var=tk.StringVar(master=self.window,value=labels[column+1] if column is not None else labels[0]);self.columns[key]=var
            editor.field(cell,key.upper()+' column',var,'Map this field to a CSV column. Preview validates rows before changing your points.',labels)
        self.role=tk.StringVar(master=self.window,value='Ground control');self.units=tk.StringVar(master=self.window,value='Metres')
        settings=tk.Frame(body,bg='#1a2024');settings.pack(fill='x')
        for title,var,values,tip in [('XYZ role',self.role,['Ground control','Check point'],'Use ground control for known reference positions, or check points for independent evaluation. Estimated RealityScan positions are not surveyed ground truth.'),
                                     ('XYZ units',self.units,['Metres','Millimetres','Centimetres','Feet'],'Positions and survey uncertainties are converted into metres. No geographic coordinate conversion is performed.')]:
            cell=tk.Frame(settings,bg='#1a2024');cell.pack(side='left',fill='x',expand=True,padx=5);editor.field(cell,title,var,tip,values)
        self.replace=tk.BooleanVar(master=self.window,value=False);self.skip=tk.BooleanVar(master=self.window,value=False)
        for text,var,tip in [('Replace existing measurements in matching images',self.replace,'Default keeps your existing measurement. Replacement affects only matching point/image pairs.'),
                             ('Skip unmatched images (import matching measurements only)',self.skip,'Unmatched image rows are counted in the preview. Nothing is silently mapped to a different image.')]:
            check=tk.Checkbutton(body,text=text,variable=var,bg='#1a2024',fg='#e7edef',selectcolor='#242d32',activebackground='#1a2024');check.pack(anchor='w');editor.guide(check,text,tip)
        from tkinter.scrolledtext import ScrolledText
        self.preview=ScrolledText(body,height=10,wrap='word',bg='#242d32',fg='#e7edef',font=('Segoe UI',10),relief='flat');self.preview.pack(fill='both',expand=True,pady=8)
        self.preview.insert('1.0','RealityScan image measurements are top-left, zero-based pixels. They must refer to the same original equirectangular images at the same resolution.\n\nA perspective/cubemap CSV cannot be copied onto panoramas. Use Point names only and re-mark features, or supply a matching native-ERP measurement export. Local XYZ is a separate import, matched by point name.\n\nPreview the import before applying it.');self.preview.configure(state='disabled')
        buttons=tk.Frame(body,bg='#1a2024');buttons.pack(fill='x')
        editor.button(buttons,'Preview import',self.preview_import,'Validate mappings, dimensions and merge counts without modifying points.').pack(side='left')
        editor.button(buttons,'Import',self.apply,'Revalidate the current settings and apply the import as one undoable edit.',primary=True).pack(side='right')
        editor.button(buttons,'Cancel',self.window.destroy).pack(side='right',padx=8)
        editor.style_dropdowns(self.window)
    def change_kind(self):
        self.candidate=None
    def preview_import(self):
        try:
            labels=['— Not used —']+[f'{i+1}: {h}' for i,h in enumerate(self.table['headers'])]
            mapping={key:labels.index(var.get())-1 if var.get()!=labels[0] else None for key,var in self.columns.items()}
            names_only=self.kind.get()=='Point names only';kind='xyz' if self.kind.get()=='Local XYZ coordinates' else 'measurements'
            if kind=='xyz' and not names_only and self.table['kind']=='measurements' and mapping.get('z') is None:
                raise ValueError('This is an image-measurement export, not an XYZ file. Export local XYZ separately or use point names only.')
            factor={'Metres':1,'Millimetres':.001,'Centimetres':.01,'Feet':.3048}[self.units.get()]
            sigma=.01/factor if kind=='xyz' else float(self.editor.settings['sigma_px'].get())
            data,stats=import_table(self.table,self.editor.controls,self.editor.image_sizes,kind,mapping,self.replace.get(),names_only,
                                    'check' if self.role.get()=='Check point' else 'gcp',factor,sigma,self.skip.get())
            self.candidate=data;message=json_summary(stats)+'\n\n'+ ('Point names imported without image measurements. Mark the same features on your panoramas before refinement.' if names_only else 'Ready to import. No source files will be modified.')
        except Exception as error: self.candidate=None;message=str(error)
        self.preview.configure(state='normal');self.preview.delete('1.0','end');self.preview.insert('1.0',message);self.preview.configure(state='disabled')
        return self.candidate is not None
    def apply(self):
        if not self.editor.can_edit() or not self.preview_import(): return
        self.editor.document.checkpoint();self.editor.controls=self.candidate
        self.editor.audit.event('csv_import',source=self.table['path'],kind=self.kind.get(),replace=self.replace.get(),skip_unmatched=self.skip.get(),units=self.units.get())
        if not self.editor.selected() and self.candidate['points']: self.editor.active.set(self.candidate['points'][0]['id'])
        self.editor.refresh();self.editor.status.set('CSV imported. Review measurements and save controls; Undo restores the previous document.');self.window.destroy()


def json_summary(stats):
    return '\n'.join(f'{key.replace("_"," ").capitalize()}: {value}' for key,value in stats.items())
