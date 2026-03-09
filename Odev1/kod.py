import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def unpickle(dosya_yolu):

    with open(dosya_yolu, 'rb') as fo:
        veri_sozlugu = pickle.load(fo, encoding='bytes')
    return veri_sozlugu

def yerel_veri_yukle(klasor_yolu, veri_siniri=None):

    print(f"'{klasor_yolu}' klasöründen veri seti yükleniyor...")
    
    X_train_list = []
    y_train_list = []
    
    for i in range(1, 6):
        dosya_yolu = os.path.join(klasor_yolu, f'data_batch_{i}')
        veri_sozlugu = unpickle(dosya_yolu)
        
        X_train_list.append(veri_sozlugu[b'data'])
        y_train_list += veri_sozlugu[b'labels']
        
    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)
    
    test_dosya_yolu = os.path.join(klasor_yolu, 'test_batch')
    test_sozlugu = unpickle(test_dosya_yolu)
    
    X_test = test_sozlugu[b'data']
    y_test = np.array(test_sozlugu[b'labels'])
    
    if veri_siniri:
        X_train = X_train[:veri_siniri]
        y_train = y_train[:veri_siniri]
        X_test = X_test[:veri_siniri // 5]
        y_test = y_test[:veri_siniri // 5]
        
    return X_train, X_test, y_train, y_test

def kullanici_girdilerini_al():

    while True:
        mesafe_secimi = input("Mesafe ölçüm yöntemini seçin (1: Manhattan, 2: Öklid): ")
        if mesafe_secimi in ['1', '2']:
            p_degeri = 1 if mesafe_secimi == '1' else 2
            mesafe_adi = "Manhattan" if p_degeri == 1 else "Öklid"
            break
        print("Hatalı giriş! Lütfen 1 veya 2 tuşlayın.")
        
    while True:
        try:
            k_degeri = int(input("Lütfen 'k' (komşu) değerini girin (örn. 3, 5, 7): "))
            if k_degeri > 0:
                break
            print("k değeri 0'dan büyük bir tam sayı olmalıdır.")
        except ValueError:
            print("Lütfen geçerli bir tam sayı girin.")
            
    return p_degeri, mesafe_adi, k_degeri

def modeli_egit_ve_test_et(X_train, X_test, y_train, y_test, p_degeri, k_degeri):

    print(f"\nModel {k_degeri}-NN ve {p_degeri} p-değeri ile eğitiliyor...")
    
    knn_model = KNeighborsClassifier(n_neighbors=k_degeri, p=p_degeri, n_jobs=-1)
    knn_model.fit(X_train, y_train)
    
    print("Eğitim seti başarı oranı hesaplanıyor...")
    train_tahminleri = knn_model.predict(X_train)
    train_basari = accuracy_score(y_train, train_tahminleri)
    
    print("Test seti başarı oranı hesaplanıyor...")
    test_tahminleri = knn_model.predict(X_test)
    test_basari = accuracy_score(y_test, test_tahminleri)
    
    return train_basari, test_basari

def main():
    print("=== Yerel CIFAR-10 K-NN Sınıflandırma Projesi ===\n")
    
    klasor_adi = "cifar-10-batches-py"
    
    X_train, X_test, y_train, y_test = yerel_veri_yukle(klasor_adi, veri_siniri=10000) 
    print(f"Veri hazır! {X_train.shape[0]} eğitim, {X_test.shape[0]} test verisi kullanılıyor.\n")
    
    p_degeri, mesafe_adi, k_degeri = kullanici_girdilerini_al()
    print(f"\n-> Seçimleriniz: Mesafe = {mesafe_adi}, K = {k_degeri}")
    
    train_basari, test_basari = modeli_egit_ve_test_et(X_train, X_test, y_train, y_test, p_degeri, k_degeri)
    
    print("\n" + "="*30)
    print("          SONUÇLAR")
    print("="*30)
    print(f"Eğitim Seti Başarı Oranı : %{train_basari * 100:.2f}")
    print(f"Test Seti Başarı Oranı   : %{test_basari * 100:.2f}")
    print("="*30)

if __name__ == "__main__":
    main()