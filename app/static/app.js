/* =========================================================
   Plant Care Card — App Logic
   ========================================================= */

(function () {
  'use strict';

  // --- DOM refs ---
  const uploadZone = document.getElementById('upload-zone');
  const uploadZoneContent = document.getElementById('upload-zone-content');
  const uploadPreview = document.getElementById('upload-preview');
  const fileInput = document.getElementById('file-input');
  const btnClear = document.getElementById('btn-clear');
  const btnClassify = document.getElementById('btn-classify');
  const btnLabel = btnClassify.querySelector('.btn-label');
  const btnSpinner = btnClassify.querySelector('.btn-spinner');
  const modeCareCard = document.getElementById('mode-care-card');

  const loadingSection = document.getElementById('loading-section');
  const loadingText = document.getElementById('loading-text');

  const errorToast = document.getElementById('error-toast');
  const errorMessage = document.getElementById('error-message');
  const errorClose = document.getElementById('error-close');

  const resultsSection = document.getElementById('results-section');
  const resultImg = document.getElementById('result-img');
  const resultName = document.getElementById('result-name');
  const resultLatin = document.getElementById('result-latin');
  const confidenceBadge = document.getElementById('confidence-badge');

  const careCard = document.getElementById('care-card');
  const careSummary = document.getElementById('care-summary');
  const identityDetails = document.getElementById('identity-details');
  const envDetails = document.getElementById('env-details');
  const waterDetails = document.getElementById('water-details');
  const growthDetails = document.getElementById('growth-details');
  const maintenanceDetails = document.getElementById('maintenance-details');
  const healthDetails = document.getElementById('health-details');

  let selectedFile = null;

  // --- Upload Handling ---
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  // Drag & drop
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(file) {
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'];
    if (!allowed.includes(file.type)) {
      showError('Please upload a valid image (JPG, PNG, or WebP).');
      return;
    }
    selectedFile = file;
    const url = URL.createObjectURL(file);
    uploadPreview.src = url;
    uploadPreview.classList.add('visible');
    uploadZoneContent.style.display = 'none';
    btnClear.disabled = false;
    btnClassify.disabled = false;
  }

  btnClear.addEventListener('click', resetUpload);

  function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadPreview.src = '';
    uploadPreview.classList.remove('visible');
    uploadZoneContent.style.display = '';
    btnClear.disabled = true;
    btnClassify.disabled = true;
    resultsSection.hidden = true;
    careCard.hidden = true;
  }

  // --- Error Toast ---
  function showError(msg) {
    errorMessage.textContent = msg;
    errorToast.hidden = false;
    setTimeout(() => { errorToast.hidden = true; }, 6000);
  }

  errorClose.addEventListener('click', () => { errorToast.hidden = true; });

  // --- API Call ---
  btnClassify.addEventListener('click', handleClassify);

  async function handleClassify() {
    if (!selectedFile) return;

    const useCareCard = modeCareCard.checked;
    const endpoint = useCareCard ? '/api/v1/plant-care' : '/api/v1/predict';

    // UI: loading state
    setLoading(true, useCareCard);
    resultsSection.hidden = true;
    careCard.hidden = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(endpoint, { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.message || `Server error (${res.status})`);
      }

      if (useCareCard) {
        renderCareCardResult(data);
      } else {
        renderPredictResult(data);
      }
    } catch (err) {
      showError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  function setLoading(on, isCareCard = false) {
    if (on) {
      btnLabel.hidden = true;
      btnSpinner.hidden = false;
      btnClassify.disabled = true;
      loadingSection.hidden = false;
      loadingText.textContent = isCareCard
        ? 'Researching & generating care card...'
        : 'Classifying your plant...';
    } else {
      btnLabel.hidden = false;
      btnSpinner.hidden = true;
      btnClassify.disabled = false;
      loadingSection.hidden = true;
    }
  }

  // --- Render: Predict-only ---
  function renderPredictResult(data) {
    const r = data.results;
    resultImg.src = uploadPreview.src;
    resultName.textContent = r.label;
    resultLatin.textContent = '';
    confidenceBadge.textContent = `${(r.confidence * 100).toFixed(1)}% confidence`;
    resultsSection.hidden = false;
    careCard.hidden = true;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // --- Render: Full Care Card ---
  function renderCareCardResult(data) {
    const cls = data.classification;
    const card = data.plant_care_card;

    // Classification header
    resultImg.src = uploadPreview.src;
    resultName.textContent = card.common_name || cls.label;
    resultLatin.textContent = card.latin_name || '';
    confidenceBadge.textContent = `${(cls.confidence * 100).toFixed(1)}% confidence`;

    // Summary
    careSummary.textContent = card.summary || '';

    // Identity block
    identityDetails.innerHTML = buildDetails([
      { label: 'Common Name', value: card.common_name },
      { label: 'Scientific Name', value: card.latin_name, italic: true },
      { label: 'Family', value: card.family },
      { label: 'Native Habitat', value: card.native_habitat },
    ]);

    // Environment block
    envDetails.innerHTML = buildDetails([
      { label: 'Lighting', value: card.lighting_conditions, tags: true, tagClass: 'care-tag--accent' },
      { label: 'Temperature', value: card.temperature_range_celsius ? `${card.temperature_range_celsius[0]}°C – ${card.temperature_range_celsius[1]}°C` : null },
      { label: 'Humidity', value: capitalize(card.humidity_level) },
      { label: 'Soil', value: card.soil_type },
      { label: 'Outdoors', value: card.outdoors },
      { label: 'Indoors', value: card.indoors },
    ]);

    // Watering block
    const waterItems = [
      { label: 'Interval', value: card.watering_interval_days ? `Every ${card.watering_interval_days} day${card.watering_interval_days > 1 ? 's' : ''}` : null },
      { label: 'Volume', value: card.watering_volume_l ? `${card.watering_volume_l} L per session` : null },
    ];
    if (card.watering_adjustments && card.watering_adjustments.length > 0) {
      card.watering_adjustments.forEach((adj, i) => {
        waterItems.push({
          label: `Adjustment ${i + 1}`,
          value: [adj.condition, adj.note].filter(Boolean).join(' — '),
        });
      });
    }
    waterDetails.innerHTML = buildDetails(waterItems);

    // Growth block
    growthDetails.innerHTML = buildDetails([
      { label: 'Growth Rate', value: capitalize(card.growth_rate) },
      { label: 'Mature Height', value: card.mature_height_cm ? `${card.mature_height_cm[0]} – ${card.mature_height_cm[1]} cm` : null },
      { label: 'Blooming Season', value: card.blooming_season, tags: true },
      { label: 'Planting Season', value: card.planting_season, tags: true },
    ]);

    // Maintenance block
    maintenanceDetails.innerHTML = buildDetails([
      { label: 'Difficulty', value: capitalize(card.difficulty_level) },
      { label: 'Fertilizer', value: card.fertilizer_needs },
      { label: 'Pruning Required', value: card.pruning_required, boolean: true },
    ]);

    // Health block
    healthDetails.innerHTML = buildDetails([
      { label: 'Toxicity', value: card.toxicity },
      { label: 'Common Pests', value: card.common_pests, tags: true },
      { label: 'Common Diseases', value: card.common_diseases, tags: true },
    ]);

    resultsSection.hidden = false;
    careCard.hidden = false;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // --- Helpers ---

  function buildDetails(items) {
    return items
      .filter((item) => item.value !== null && item.value !== undefined && item.value !== '')
      .map((item) => {
        let valueHtml;

        if (item.boolean !== undefined) {
          const yes = !!item.value;
          valueHtml = `<span class="care-boolean"><span class="dot ${yes ? 'dot--yes' : 'dot--no'}"></span> ${yes ? 'Yes' : 'No'}</span>`;
        } else if (item.tags && Array.isArray(item.value)) {
          valueHtml = item.value
            .map((v) => `<span class="care-tag ${item.tagClass || ''}">${formatTag(v)}</span>`)
            .join(' ');
        } else {
          valueHtml = `<span ${item.italic ? 'style="font-style: italic;"' : ''}>${escapeHtml(String(item.value))}</span>`;
        }

        return `
          <div class="care-detail">
            <span class="care-detail-label">${escapeHtml(item.label)}</span>
            <span class="care-detail-value">${valueHtml}</span>
          </div>`;
      })
      .join('');
  }

  function formatTag(val) {
    return escapeHtml(String(val).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()));
  }

  function capitalize(str) {
    if (!str) return '';
    return String(str).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
})();
