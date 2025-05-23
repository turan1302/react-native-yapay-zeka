import React, { useEffect, useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import '@tensorflow/tfjs-backend-webgl';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

export default function App() {
  const [isTfReady, setIsTfReady] = useState(false);
  const [model, setModel] = useState(null);
  const [boy, setBoy] = useState('');
  const [tahmin, setTahmin] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function loadModel() {
      try {
        await tf.ready();
        await tf.setBackend('rn-webgl');
        setIsTfReady(true);

        const modelJson = require('./assets/kilo-model/model.json');

        const model = await tf.loadLayersModel(
            bundleResourceIO(modelJson, require('./assets/kilo-model/weights.bin'))
        );

        console.log('Model başarıyla yüklendi');
        setModel(model);
      } catch (err) {
        console.error('Model yükleme hatası:', err);
      }
    }

    loadModel();
  }, []);

  const predictKilo = async () => {
    if (!model) {
      alert('Model henüz yüklenmedi!');
      return;
    }
    if (!boy || isNaN(parseFloat(boy))) {
      alert('Lütfen geçerli bir boy giriniz!');
      return;
    }

    setLoading(true);

    try {
      const boyMetre = parseFloat(boy) / 100;
      const inputTensor = tf.tensor2d([boyMetre], [1, 1], 'float32');
      const outputTensor = model.predict(inputTensor);
      const outputData = await outputTensor.data(); // Async data alma
      setTahmin(outputData[0].toFixed(2));
    } catch (error) {
      alert('Tahmin sırasında hata oluştu: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
      <View style={styles.container}>
        <Text style={{ fontSize: 18, marginBottom: 15 }}>TensorFlow.js React Native Boy-Kilo Tahmini</Text>

        {!isTfReady && <Text>TensorFlow Yükleniyor...</Text>}

        <TextInput
            style={styles.input}
            placeholder="Boy (cm olarak giriniz)"
            keyboardType="numeric"
            value={boy}
            onChangeText={setBoy}
        />

        <Button title={loading ? 'Tahmin Yapılıyor...' : 'Kilo Tahmin Et'} onPress={predictKilo} disabled={loading} />

        {tahmin !== null && <Text style={{ marginTop: 20, fontSize: 16 }}>Tahmini Kilo: {tahmin} kg</Text>}
      </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', padding: 20 },
  input: { borderWidth: 1, borderColor: '#999', padding: 10, marginVertical: 10, borderRadius: 5 },
});
