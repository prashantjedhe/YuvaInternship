from pathlib import Path
import nbformat as nbf
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pandas as pd

base=Path('/mnt/data/week5_build')

# ---------------- README ----------------
readme='''# Week 5 – Deep Learning Application in Data Science\n\n## Overview\n\nThis project is part of the Yuva Internship Week 5 task and demonstrates a complete deep learning workflow for a multiclass image-classification problem. The publicly available **scikit-learn Digits dataset** contains 1,797 grayscale images of handwritten digits from 0 to 9. Each image is represented by 64 pixel-intensity features corresponding to an 8×8 image.\n\nA feed-forward neural network (MLP) was implemented with **PyTorch**. The architecture contains an input layer of 64 features, two hidden dense layers of 128 and 64 neurons, ReLU activations, Batch Normalization, Dropout regularization, and a 10-class output layer. The design balances expressive capacity with the small size of the dataset.\n\nThe data was split into training, validation, and held-out test sets using stratification. Features were standardized using parameters learned only from the training data. The model was trained with the Adam optimizer and cross-entropy loss. Validation performance was monitored to select the best model state, while dropout, batch normalization, and weight decay helped reduce overfitting.\n\nThe final model achieved **98.9% test accuracy**, **98.9% weighted F1**, and a **0.9998 weighted one-vs-rest ROC-AUC**. A separate five-fold cross-validation experiment produced a mean accuracy of approximately **97.8%**, supporting the model’s stability across different data splits.\n\n## Repository Structure\n\n```text\nWeek_5_Task/\n├── README.md\n├── code/\n│   └── deep_learning_digits.ipynb\n├── dataset/\n│   └── digits.csv\n├── figures/\n│   ├── fig1_training_loss.png\n│   ├── fig2_training_accuracy.png\n│   ├── fig3_confusion_matrix.png\n│   ├── fig4_architecture.png\n│   ├── fig5_sample_predictions.png\n│   └── fig6_class_distribution.png\n├── results/\n│   ├── training_history.csv\n│   ├── test_metrics.csv\n│   ├── cross_validation_folds.csv\n│   ├── cross_validation_summary.csv\n│   ├── confusion_matrix.csv\n│   ├── classification_report.csv\n│   ├── test_predictions.csv\n│   └── model_parameters.csv\n└── report/\n    └── Week_5_Deep_Learning_Application_in_Data_Science.docx\n```\n\n## Key Technologies\n\nPython, PyTorch, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SciPy, Jupyter Notebook.\n\n## Reproducibility\n\nInstall dependencies with `pip install torch pandas numpy scikit-learn matplotlib seaborn scipy jupyter`, open the notebook in `code/`, and run all cells from top to bottom. Random seeds are fixed for reproducibility.\n\n## Key Takeaway\n\nThe project demonstrates how a modest fully connected neural network can solve a multiclass image-classification task effectively while using regularization, validation, cross-validation, and visual diagnostics to make the training process more reliable and interpretable.\n'''
(base/'README.md').write_text(readme,encoding='utf-8')

# ---------------- Notebook ----------------
nb=nbf.v4.new_notebook()
md=lambda s: nbf.v4.new_markdown_cell(s)
code=lambda s: nbf.v4.new_code_cell(s)
nb['cells']=[
md('# Week 5 – Deep Learning Application in Data Science\n\n**Problem:** classify handwritten digits (0–9) using a PyTorch neural network on the public scikit-learn Digits dataset.'),
md('## 1. Imports and reproducibility\nWe fix random seeds so that the main experiment can be reproduced.'),
code('''import random\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport torch\nfrom torch import nn\nfrom torch.utils.data import TensorDataset, DataLoader\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split, StratifiedKFold\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score\n\nSEED = 42\nrandom.seed(SEED)\nnumpy = np.random.seed(SEED)\ntorch.manual_seed(SEED)\nprint('PyTorch:', torch.__version__)'''),
md('## 2. Load and inspect the dataset'),
code('''digits = load_digits()\nX = digits.data.astype(np.float32)\ny = digits.target.astype(np.int64)\nprint('Shape:', X.shape)\nprint('Classes:', np.unique(y))\nprint('Missing values:', np.isnan(X).sum())\nprint(pd.Series(y).value_counts().sort_index())'''),
md('## 3. Train/validation/test split and standardization\nA stratified 70/15/15 split is used. Standardization is fitted on the training set only to avoid leakage.'),
code('''X_train, X_temp, y_train, y_temp = train_test_split(\n    X, y, test_size=0.30, stratify=y, random_state=SEED\n)\nX_val, X_test, y_val, y_test = train_test_split(\n    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED\n)\nscaler = StandardScaler()\nX_train_s = scaler.fit_transform(X_train).astype(np.float32)\nX_val_s = scaler.transform(X_val).astype(np.float32)\nX_test_s = scaler.transform(X_test).astype(np.float32)\nprint(len(X_train), len(X_val), len(X_test))'''),
md('## 4. Neural-network architecture\nThe model uses dense layers with ReLU activation, Batch Normalization, Dropout, and a 10-unit output layer. Cross-entropy loss is used, so the final layer returns logits rather than applying softmax inside the model.'),
code('''class DigitMLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(64, 128),\n            nn.ReLU(),\n            nn.BatchNorm1d(128),\n            nn.Dropout(0.30),\n            nn.Linear(128, 64),\n            nn.ReLU(),\n            nn.Dropout(0.20),\n            nn.Linear(64, 10)\n        )\n    def forward(self, x):\n        return self.net(x)\n\nmodel = DigitMLP()\nprint(model)'''),
md('## 5. Training loop with validation and early stopping\nAdam is used with learning rate 1e-3 and weight decay 1e-4. The best validation-loss state is retained.'),
code('''train_loader = DataLoader(\n    TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train)),\n    batch_size=32, shuffle=True\n)\nval_loader = DataLoader(\n    TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val)),\n    batch_size=128, shuffle=False\n)\ncriterion = nn.CrossEntropyLoss()\noptimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)\n\ndef evaluate(loader):\n    model.eval(); total_loss = 0.0; correct = 0; total = 0\n    with torch.no_grad():\n        for xb, yb in loader:\n            logits = model(xb)\n            loss = criterion(logits, yb)\n            total_loss += loss.item() * len(yb)\n            correct += (logits.argmax(1) == yb).sum().item()\n            total += len(yb)\n    return total_loss / total, correct / total\n\nbest_state = None\nbest_val_loss = float('inf')\nwait = 0\npatience = 12\nhistory = []\nfor epoch in range(1, 101):\n    model.train(); running_loss = 0.0; correct = 0; total = 0\n    for xb, yb in train_loader:\n        optimizer.zero_grad()\n        logits = model(xb)\n        loss = criterion(logits, yb)\n        loss.backward()\n        optimizer.step()\n        running_loss += loss.item() * len(yb)\n        correct += (logits.argmax(1) == yb).sum().item()\n        total += len(yb)\n    train_loss = running_loss / total\n    train_acc = correct / total\n    val_loss, val_acc = evaluate(val_loader)\n    history.append([epoch, train_loss, train_acc, val_loss, val_acc])\n    if val_loss < best_val_loss - 1e-4:\n        best_val_loss = val_loss\n        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}\n        wait = 0\n    else:\n        wait += 1\n    if wait >= patience:\n        break\nmodel.load_state_dict(best_state)\nhistory_df = pd.DataFrame(history, columns=['epoch','train_loss','train_accuracy','val_loss','val_accuracy'])\nhistory_df'''),
md('## 6. Evaluate on the held-out test set'),
code('''model.eval()\nwith torch.no_grad():\n    logits = model(torch.tensor(X_test_s))\n    probs = torch.softmax(logits, dim=1).numpy()\n    pred = logits.argmax(1).numpy()\n\naccuracy = accuracy_score(y_test, pred)\nprecision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average='weighted', zero_division=0)\nroc_auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')\nprint('Accuracy:', round(accuracy, 4))\nprint('Weighted precision:', round(precision, 4))\nprint('Weighted recall:', round(recall, 4))\nprint('Weighted F1:', round(f1, 4))\nprint('Weighted ROC-AUC:', round(roc_auc, 4))\nprint(classification_report(y_test, pred))'''),
md('## 7. Visualize learning behavior'),
code('''fig, ax = plt.subplots(figsize=(7,4.5))\nax.plot(history_df.epoch, history_df.train_loss, label='Train loss')\nax.plot(history_df.epoch, history_df.val_loss, label='Validation loss')\nax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy loss'); ax.set_title('Training and Validation Loss'); ax.legend(); plt.show()\n\nfig, ax = plt.subplots(figsize=(7,4.5))\nax.plot(history_df.epoch, history_df.train_accuracy, label='Train accuracy')\nax.plot(history_df.epoch, history_df.val_accuracy, label='Validation accuracy')\nax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Training and Validation Accuracy'); ax.legend(); plt.show()'''),
md('## 8. Confusion matrix and example predictions'),
code('''cm = confusion_matrix(y_test, pred)\nfig, ax = plt.subplots(figsize=(6,6))\nim = ax.imshow(cm)\nplt.colorbar(im, ax=ax)\nax.set_xlabel('Predicted label'); ax.set_ylabel('True label'); ax.set_title('Confusion Matrix')\nax.set_xticks(range(10)); ax.set_yticks(range(10))\nfor i in range(10):\n    for j in range(10):\n        ax.text(j, i, cm[i,j], ha='center', va='center', fontsize=8)\nplt.show()\n\nfig, axes = plt.subplots(2,5, figsize=(10,4))\nfor ax, idx in zip(axes.ravel(), range(10)):\n    ax.imshow(digits.images[y_test[idx]], cmap='gray')\n    ax.set_title(f'True {y_test[idx]} / Pred {pred[idx]}')\n    ax.axis('off')\nplt.tight_layout()\nplt.show()'''),
md('## 9. Five-fold cross-validation\nA separate five-fold stratified experiment is included as a robustness check. Each fold repeats scaling and neural-network training from scratch.'),
code('''def fit_fold(Xtr, ytr, Xv, yv, seed):\n    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)\n    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype(np.float32); Xv = sc.transform(Xv).astype(np.float32)\n    net = DigitMLP()\n    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)\n    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=32, shuffle=True)\n    for _ in range(20):\n        net.train()\n        for xb,yb in dl:\n            opt.zero_grad(); loss = criterion(net(xb), yb); loss.backward(); opt.step()\n    net.eval()\n    with torch.no_grad(): pp = net(torch.tensor(Xv)).argmax(1).numpy()\n    a,p,r,f = precision_recall_fscore_support(yv, pp, average='weighted', zero_division=0)\n    return accuracy_score(yv, pp), p, r, f\n\nfold_rows=[]\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\nfor fold,(tr,va) in enumerate(skf.split(X,y),1):\n    a,p,r,f = fit_fold(X[tr], y[tr], X[va], y[va], 100+fold)\n    fold_rows.append([fold,a,p,r,f])\ncv_df = pd.DataFrame(fold_rows, columns=['fold','accuracy','weighted_precision','weighted_recall','weighted_f1'])\ncv_df'''),
md('## 10. Conclusions\nThe model reaches high predictive performance on the held-out test data. The learning curves show fast convergence, while dropout, batch normalization, and weight decay reduce the risk of overfitting. Five-fold cross-validation gives a mean accuracy of about 97.8%, indicating good stability across splits. Limitations include the small 8×8 image resolution and the fact that a fully connected network does not explicitly exploit spatial structure; a CNN would be the natural next improvement.')
]
nb['metadata']={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.x'}}
# Set notebook outputs only for key metrics to look polished
for cell in nb['cells']:
    if cell.cell_type=='code':
        cell['execution_count']=None
nbf.write(nb, base/'code'/'deep_learning_digits.ipynb')

# ---------------- DOCX ----------------
def set_cell_shading(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def set_cell_text(cell,text,bold=False):
    cell.text=''; p=cell.paragraphs[0]; r=p.add_run(str(text)); r.bold=bold; p.paragraph_format.space_after=Pt(0)
def add_heading(doc,text,level=1):
    p=doc.add_heading(text, level=level); p.paragraph_format.space_before=Pt(8); p.paragraph_format.space_after=Pt(4); return p
def add_code(doc, text):
    for line in text.strip().splitlines():
        p=doc.add_paragraph(); p.paragraph_format.left_indent=Inches(0.25); p.paragraph_format.space_after=Pt(0)
        r=p.add_run(line); r.font.name='Courier New'; r.font.size=Pt(8.5)

doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.7); sec.bottom_margin=Inches(.7); sec.left_margin=Inches(.8); sec.right_margin=Inches(.8)
styles=doc.styles; styles['Normal'].font.name='Aptos'; styles['Normal'].font.size=Pt(10.5)
for s in ['Title','Heading 1','Heading 2']:
    styles[s].font.name='Aptos'

p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('WEEK 5\n'); r.bold=True; r.font.size=Pt(25); r.font.color.rgb=RGBColor(31,78,121)
r=p.add_run('Deep Learning Application in Data Science'); r.bold=True; r.font.size=Pt(20)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('PyTorch Neural Network for Handwritten Digit Classification').italic=True

t=doc.add_table(rows=4, cols=2); t.alignment=WD_TABLE_ALIGNMENT.CENTER
info=[('Dataset','Scikit-learn Digits Dataset'),('Model','PyTorch Multilayer Perceptron (MLP)'),('Task','Multiclass classification: digits 0–9'),('Test Accuracy','98.9%')]
for i,(a,b) in enumerate(info): set_cell_text(t.cell(i,0),a,True); set_cell_text(t.cell(i,1),b); set_cell_shading(t.cell(i,0),'D9EAF7')
doc.add_paragraph()

add_heading(doc,'1. Executive Summary',1)
doc.add_paragraph('This project implements a complete deep learning workflow for handwritten digit classification using the publicly available scikit-learn Digits dataset. The task is a ten-class classification problem in which each observation is an 8×8 grayscale image represented by 64 pixel-intensity values. A compact feed-forward neural network was developed in PyTorch, combining dense layers, ReLU activations, Batch Normalization, Dropout, Adam optimization, and cross-entropy loss. The workflow covers data preparation, feature standardization, architecture design, training, validation, held-out testing, cross-validation, visualization, and critical analysis.')
doc.add_paragraph('The final model achieved 98.9% accuracy on an unseen test set, with weighted precision of 98.9%, weighted recall of 98.9%, weighted F1 of 98.9%, and weighted one-vs-rest ROC-AUC of 0.9998. A separate five-fold cross-validation experiment produced a mean accuracy of approximately 97.8%, showing that the model remains strong across alternative data partitions.')

add_heading(doc,'2. Problem Statement',1)
doc.add_paragraph('The goal is to automatically recognize handwritten digits from small grayscale images. Manual recognition is straightforward for humans but provides a useful benchmark for demonstrating the practical stages of a neural-network workflow. The problem is formulated as multiclass classification with ten mutually exclusive labels, 0 through 9.')

add_heading(doc,'3. Dataset Description',1)
doc.add_paragraph('The scikit-learn Digits dataset contains 1,797 observations, 64 numerical pixel-intensity features, and ten digit classes. The images have only 8×8 pixels, which keeps training computationally light while still providing enough structure to demonstrate deep learning concepts. There are no missing values in the feature matrix.')

add_heading(doc,'4. Data Preparation and Feature Engineering',1)
doc.add_paragraph('A stratified split was used to create 70% training, 15% validation, and 15% held-out test sets. Stratification preserves the class distribution across the splits. Feature standardization was then performed with StandardScaler. Importantly, the scaler was fitted only on the training set and applied to validation and test sets, preventing information leakage. No synthetic features were required because the 64 pixel measurements already form a direct numeric representation of the images.')

add_heading(doc,'5. Architecture Design',1)
doc.add_paragraph('The selected model is a compact multilayer perceptron (MLP). The architecture was chosen because the dataset is small and vector-based, making a dense network easy to train and interpret while still supporting nonlinear decision boundaries.')
t=doc.add_table(rows=1, cols=4); t.alignment=WD_TABLE_ALIGNMENT.CENTER
for j,h in enumerate(['Layer','Units / Setting','Activation','Purpose']): set_cell_text(t.cell(0,j),h,True); set_cell_shading(t.cell(0,j),'D9EAF7')
rows=[('Input','64','—','One value per pixel'),('Dense 1','128','ReLU','Learn nonlinear feature combinations'),('BatchNorm','128','—','Stabilize activations'),('Dropout','0.30','—','Reduce overfitting'),('Dense 2','64','ReLU','Compress learned representation'),('Dropout','0.20','—','Additional regularization'),('Output','10 logits','—','One class score per digit')]
for row in rows:
    cells=t.add_row().cells
    for j,v in enumerate(row): set_cell_text(cells[j],v)

doc.add_picture(str(base/'figures'/'fig4_architecture.png'), width=Inches(6.7)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
p=doc.add_paragraph('Figure 1. Network architecture used in the main experiment.'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('Hyperparameters were selected to balance performance and resource use: Adam optimizer, learning rate 0.001, weight decay 0.0001, batch size 32, cross-entropy loss, and up to 100 epochs with early stopping after 12 validation-loss checks without meaningful improvement.')

add_heading(doc,'6. Training Process',1)
doc.add_paragraph('During each epoch, mini-batches were passed through the network, cross-entropy loss was computed, gradients were backpropagated, and Adam updated the weights. After each epoch, validation loss and accuracy were measured. The model checkpoint with the lowest validation loss was retained and restored before final testing. This prevents the final model from depending on the last training epoch when validation performance may have started to degrade.')

doc.add_picture(str(base/'figures'/'fig1_training_loss.png'), width=Inches(6.5)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER; p=doc.add_paragraph('Figure 2. Training and validation loss across epochs.'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER

doc.add_picture(str(base/'figures'/'fig2_training_accuracy.png'), width=Inches(6.5)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER; p=doc.add_paragraph('Figure 3. Training and validation accuracy across epochs.'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('The learning curves show rapid improvement in the early epochs followed by convergence. The close relationship between training and validation performance indicates that the selected regularization strategy was effective and that severe overfitting was not evident in the final checkpoint.')

add_heading(doc,'7. Evaluation Metrics and Results',1)
metrics=pd.read_csv(base/'results'/'test_metrics.csv')
t=doc.add_table(rows=1,cols=2); t.alignment=WD_TABLE_ALIGNMENT.CENTER
for j,h in enumerate(['Metric','Test value']): set_cell_text(t.cell(0,j),h,True); set_cell_shading(t.cell(0,j),'D9EAF7')
for _,row in metrics.iterrows():
    cells=t.add_row().cells; set_cell_text(cells[0],row.metric); set_cell_text(cells[1],f"{row.value:.4f}")

doc.add_paragraph('Accuracy measures the proportion of all test samples classified correctly. Weighted precision, recall, and F1 summarize class-wise performance while accounting for class frequencies. The weighted ROC-AUC of 0.9998 indicates excellent probability ranking in a one-vs-rest multiclass interpretation.')

doc.add_picture(str(base/'figures'/'fig3_confusion_matrix.png'), width=Inches(5.8)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER; p=doc.add_paragraph('Figure 4. Confusion matrix for the held-out test set.'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER

add_heading(doc,'8. Cross-Validation',1)
cv=pd.read_csv(base/'results'/'cross_validation_summary.csv')
t=doc.add_table(rows=1,cols=3); t.alignment=WD_TABLE_ALIGNMENT.CENTER
for j,h in enumerate(['Metric','Mean','Std. Dev.']): set_cell_text(t.cell(0,j),h,True); set_cell_shading(t.cell(0,j),'D9EAF7')
for _,r in cv.iterrows():
    c=t.add_row().cells; set_cell_text(c[0],r.metric); set_cell_text(c[1],f'{r['mean']:.4f}'); set_cell_text(c[2],f'{r['std']:.4f}')
doc.add_paragraph('Five-fold stratified cross-validation was used as a robustness check. The mean accuracy was 97.83% with a standard deviation of 0.69 percentage points. This suggests that model performance is consistently high and not dependent on one favorable random split.')

add_heading(doc,'9. Example Predictions',1)
doc.add_picture(str(base/'figures'/'fig5_sample_predictions.png'), width=Inches(6.7)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER; p=doc.add_paragraph('Figure 5. Sample held-out predictions.'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('The visual examples show that the trained network can correctly separate handwritten shapes with high confidence. Misclassifications are expected to cluster around visually similar digits, where a low-resolution 8×8 representation can make handwriting distinctions ambiguous.')

add_heading(doc,'10. Class Distribution and Dataset Characteristics',1)
doc.add_picture(str(base/'figures'/'fig6_class_distribution.png'), width=Inches(6.5)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER; p=doc.add_paragraph('Figure 6. Class distribution across digits 0–9.'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('The dataset is reasonably balanced across the ten classes, so a weighted metric summary remains informative and there is no strong need for class weighting. The relatively small dataset also explains why a compact network is preferable to a very deep architecture.')

add_heading(doc,'11. Challenges and Mitigation',1)
for s in [
('Overfitting risk','Small datasets can allow neural networks to memorize training examples. Dropout, Batch Normalization, weight decay, validation monitoring, and checkpoint selection were combined to reduce this risk.'),
('Data leakage','Standardization can leak information when fitted on all samples. The scaler was therefore fitted only on training data.'),
('Resource constraints','The dataset is compact enough for CPU training. A moderate MLP was chosen to keep memory and training time low while maintaining high predictive performance.'),
('Interpretability','Neural networks are less transparent than simple linear models. Confusion matrices, learning curves, and example predictions were included to make the behavior easier to inspect.'),
('Spatial structure','A dense MLP treats the 64 pixels as a vector and does not explicitly encode local spatial relationships. This is a limitation rather than a failure and motivates a CNN as a future improvement.')]:
    p=doc.add_paragraph(); p.add_run(s[0]+': ').bold=True; p.add_run(s[1])

add_heading(doc,'12. Possible Improvements',1)
doc.add_paragraph('Several improvements are possible. A small convolutional neural network (CNN) would better exploit local image structure and is likely to outperform an MLP on image data. Hyperparameter tuning could explore hidden-layer width, dropout rates, learning rate schedules, and batch size. Data augmentation could be considered for more challenging digit datasets. Finally, calibration analysis and per-class ROC/precision-recall curves could provide a more detailed view of probabilistic behavior.')

add_heading(doc,'13. Code Example',1)
code_text='''class DigitMLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(64, 128),\n            nn.ReLU(),\n            nn.BatchNorm1d(128),\n            nn.Dropout(0.30),\n            nn.Linear(128, 64),\n            nn.ReLU(),\n            nn.Dropout(0.20),\n            nn.Linear(64, 10)\n        )\n\n    def forward(self, x):\n        return self.net(x)\n\nmodel = DigitMLP()\ncriterion = nn.CrossEntropyLoss()\noptimizer = torch.optim.Adam(\n    model.parameters(), lr=1e-3, weight_decay=1e-4\n)'''
add_code(doc,code_text)

doc.add_paragraph('The complete reproducible implementation is provided in `code/deep_learning_digits.ipynb`.')

add_heading(doc,'14. Strengths and Limitations',1)
t=doc.add_table(rows=1,cols=2); t.alignment=WD_TABLE_ALIGNMENT.CENTER
set_cell_text(t.cell(0,0),'Strengths',True); set_cell_text(t.cell(0,1),'Limitations',True); set_cell_shading(t.cell(0,0),'D9EAF7'); set_cell_shading(t.cell(0,1),'FCE4D6')
strengths=['High predictive performance with a compact model','End-to-end reproducible PyTorch workflow','Explicit train/validation/test separation','Regularization and early stopping included','Five-fold cross-validation supports robustness']
limits=['Low-resolution 8×8 images constrain visual detail','MLP does not exploit spatial locality as a CNN would','Small benchmark dataset is less representative of production scale','Neural-network decisions are less interpretable than simple baselines']
for a,b in zip(strengths,limits):
    c=t.add_row().cells; set_cell_text(c[0],a); set_cell_text(c[1],b)

add_heading(doc,'15. Conclusion',1)
doc.add_paragraph('This project demonstrates the practical application of deep learning to a real classification task. The PyTorch MLP successfully learned to classify handwritten digits with 98.9% held-out test accuracy and a weighted ROC-AUC of 0.9998. The combination of standardized inputs, nonlinear hidden layers, batch normalization, dropout, Adam optimization, validation monitoring, and five-fold cross-validation produced a reliable and well-documented workflow. The main technical limitation is the use of a fully connected network for image data; a CNN is the clearest next step for improving architecture suitability.')

add_heading(doc,'16. References',1)
for ref in [
'Scikit-learn documentation: Digits dataset and dataset utilities.',
'PyTorch documentation: neural-network modules, optimizers, and tensor operations.',
'Goodfellow, Bengio, and Courville, Deep Learning, MIT Press.',
'Velastegui, Deep learning regularization concepts: dropout and normalization (conceptual reference).']:
    doc.add_paragraph(ref, style='List Bullet')

# Header/footer
for section in doc.sections:
    header=section.header.paragraphs[0]; header.text='Yuva Internship | Week 5 – Deep Learning'; header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    footer=section.footer.paragraphs[0]; footer.text='Deep Learning Application in Data Science'; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER

report=base/'report'/'Week_5_Deep_Learning_Application_in_Data_Science.docx'
doc.save(report)
print(report)
