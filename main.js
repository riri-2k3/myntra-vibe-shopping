 // FIXED: Updated backend URL configuration
    const BACKEND_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
      ? 'http://localhost:8000' 
      : 'http://localhost:8000'; 

    // Helper function to escape HTML
    function escapeHtml(unsafe) {
      if (!unsafe) return '';
      return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    // Show error message
    function showError(message, containerId = 'resultsGrid') {
      const container = document.getElementById(containerId);
      container.innerHTML = `<div class="error-message">⚠️ ${message}</div>`;
    }

    // Show success message
    function showSuccess(message, containerId = 'resultsGrid') {
      const container = document.getElementById(containerId);
      container.innerHTML = `<div class="success-message">✅ ${message}</div>`;
    }

    // Show loading spinner
    function showLoading(containerId = 'resultsGrid') {
      const container = document.getElementById(containerId);
      container.innerHTML = '<div class="loading-spinner" style="display:block;"></div>';
    }

    // FIXED: Render product cards from a list of products
    function renderProducts(products, containerId = 'resultsGrid') {
      const grid = document.getElementById(containerId);
      grid.innerHTML = ''; 

      if (!products || products.length === 0) {
        grid.innerHTML = '<div style="text-align:center; padding: 20px; color: #666;">No products found for this search. Try a different query!</div>';
        return;
      }

      products.forEach(p => {
        const card = document.createElement('div');
        card.className = 'product';
        
        const imageUrl = p.image_url || p.image || 'https://placehold.co/400x400/f5f5f5/999?text=No+Image';
        
        let vibeTagsText = '';
        if (p.vibe_tags) {
          vibeTagsText = Array.isArray(p.vibe_tags) ? p.vibe_tags.join(', ') : p.vibe_tags;
        }
        
        card.innerHTML = `
          <div class="imgwrap">
            <img src="${imageUrl}" alt="${escapeHtml(p.title || 'Product')}" onerror="this.src='https://placehold.co/400x400/f5f5f5/999?text=No+Image'" />
          </div>
          <div class="meta">
            <h4>${escapeHtml(p.title || 'Untitled Product')}</h4>
            <div class="price">₹${p.price || 'N/A'}</div>
            <div class="tags">${escapeHtml(vibeTagsText)}</div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    // FIXED: Render featured product cards
    function renderFeatured(products, containerId = 'featuredRow') {
      const row = document.getElementById(containerId);
      row.innerHTML = '';
      
      if (!products || products.length === 0) {
        row.innerHTML = '<div style="padding: 20px; color: #666;">No featured products available</div>';
        return;
      }
      
      products.forEach(p => {
        const el = document.createElement('div');
        el.className = 'featured-item';
        
        const imageUrl = p.image_url || p.image || 'https://placehold.co/180x140/f5f5f5/999?text=No+Image';
        
        el.innerHTML = `
          <img src="${imageUrl}" alt="${escapeHtml(p.title || 'Product')}" onerror="this.src='https://placehold.co/180x140/f5f5f5/999?text=No+Image'" />
          <div class="ft-meta">
            <div style="font-weight:700">${escapeHtml(p.title || 'Product')}</div>
            <div class="small muted">₹${p.price || 'N/A'}</div>
          </div>
        `;
        row.appendChild(el);
      });
    }

    // Render a list of chips
    function renderChips(chips) {
      const area = document.getElementById('chipsArea');
      area.innerHTML = '';
      
      if (!chips || chips.length === 0) {
        area.innerHTML = '<div class="muted">No trending vibes available</div>';
        return;
      }
      
      const wrap = document.createElement('div');
      wrap.className = 'chips';
      
      chips.forEach(c => {
        const chip = document.createElement('div');
        chip.className = 'chip';
        chip.textContent = '#' + c;
        chip.onclick = () => {
          document.getElementById('vibeInput').value = c;
          submitVibe();
        };
        wrap.appendChild(chip);
      });
      area.appendChild(wrap);
    }

    // NEW: Render event chips
    function renderEventChips(events) {
      const area = document.getElementById('eventChips');
      area.innerHTML = '';
      
      if (!events || events.length === 0) {
        return;
      }
      
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.gap = '8px';
      wrap.style.flexWrap = 'wrap';
      wrap.style.marginBottom = '12px';
      
      events.forEach(event => {
        const chip = document.createElement('div');
        chip.className = 'event-chip';
        chip.textContent = `🎭 ${event.name || event}`;
        chip.onclick = () => {
          // When clicking an event chip, search for that event style
          if (event.style_keywords && event.style_keywords.length > 0) {
            document.getElementById('vibeInput').value = event.style_keywords.join(' ');
            submitVibe();
          }
        };
        wrap.appendChild(chip);
      });
      area.appendChild(wrap);
    }

    // NEW: Location-based event search
    async function searchLocationEvents() {
      const citySelect = document.getElementById('citySelect');
      const location = citySelect.value;
      
      const btn = document.getElementById('eventSearchBtn');
      const prevText = btn.innerHTML;
      btn.innerHTML = '🔍 Analyzing...'; 
      btn.disabled = true;

      try {
        console.log('📍 Searching for location events:', location);
        
        const response = await fetch(BACKEND_URL + '/search/location-events', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            location: location,
            max_results: 6
          })
        });
        
        if (!response.ok) {
          throw new Error(`Backend responded with status: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('✅ Received location event results:', data);
        
        // Show event information
        const eventInfo = document.getElementById('eventInfo');
        const eventsList = document.getElementById('eventsList');
        const currentCity = document.getElementById('currentCity');
        
        currentCity.textContent = location;
        
        if (data.events && data.events.length > 0) {
          eventInfo.style.display = 'block';
          eventsList.innerHTML = '';
          
          data.events.slice(0, 3).forEach(event => {
            const eventDiv = document.createElement('div');
            eventDiv.style.marginBottom = '8px';
            eventDiv.innerHTML = `
              <div class="event-desc">
                <strong>${event.name}</strong> ${event.type ? `(${event.type})` : ''}
              </div>
              <div style="font-size: 11px; color: #9ca3af;">
                ${event.description || 'Local cultural event'}
              </div>
            `;
            eventsList.appendChild(eventDiv);
          });
          
          // Render event chips
          renderEventChips(data.events.slice(0, 4));
        } else {
          eventInfo.style.display = 'none';
        }
        
        // Show seasonal note if available
        if (data.seasonal_note) {
          const seasonalDiv = document.createElement('div');
          seasonalDiv.style.fontSize = '12px';
          seasonalDiv.style.color = '#6b7280';
          seasonalDiv.style.marginTop = '8px';
          seasonalDiv.style.fontStyle = 'italic';
          seasonalDiv.innerHTML = `💡 ${data.seasonal_note}`;
          eventsList.appendChild(seasonalDiv);
        }
        
        if (data && data.products && data.products.length > 0) {
          // Show products in a grid format within the trend radar section
const trendCard = document.querySelector('.trend-card');
let resultsContainer = document.getElementById('trendResults');
if (!resultsContainer) {
  resultsContainer = document.createElement('div');
  resultsContainer.id = 'trendResults';
  resultsContainer.innerHTML = '<div class="section-title" style="font-size:16px; margin-top:18px;">Event Products</div><div id="trendGrid" class="grid"></div>';
  trendCard.appendChild(resultsContainer);
}
renderProducts(data.products, 'trendGrid');
          console.log(`🎭 Found ${data.products.length} products for events in ${location}`);
        } else {
          let resultsContainer = document.getElementById('trendResults');
if (!resultsContainer) {
  const trendCard = document.querySelector('.trend-card');
  resultsContainer = document.createElement('div');
  resultsContainer.id = 'trendResults';
  trendCard.appendChild(resultsContainer);
}
resultsContainer.innerHTML = '<div style="padding: 20px; color: #666; text-align: center; font-size: 13px;">No suitable products found for events in this location.</div>';
        }
        
      } catch (error) {
        console.error('❌ Error during location event search:', error);
        
        // Hide event info on error
        document.getElementById('eventInfo').style.display = 'none';
        document.getElementById('eventChips').innerHTML = '';
        
        if (error.message.includes('Failed to fetch')) {
          let resultsContainer = document.getElementById('trendResults');
if (!resultsContainer) {
  const trendCard = document.querySelector('.trend-card');
  resultsContainer = document.createElement('div');
  resultsContainer.id = 'trendResults';
  trendCard.appendChild(resultsContainer);
}
resultsContainer.innerHTML = '<div style="padding: 16px; color: #ef4444; text-align: center; font-size: 13px;">Cannot connect to server. Please check if backend is running.</div>';
        } else {
          let resultsContainer = document.getElementById('trendResults');
  if (!resultsContainer) {
    const trendCard = document.querySelector('.trend-card');
    resultsContainer = document.createElement('div');
    resultsContainer.id = 'trendResults';
    trendCard.appendChild(resultsContainer);
  }
  resultsContainer.innerHTML = '<div style="padding: 16px; color: #ef4444; text-align: center; font-size: 13px;">Event search failed. Please try again.</div>';
}
      } finally {
        btn.innerHTML = prevText; 
        btn.disabled = false;
      }
    }

    // FIXED: Vibe search function with better error handling
    async function submitVibe() {
      const vibeText = document.getElementById('vibeInput').value.trim();
      if (!vibeText) {
        showError('Please enter a vibe to search (e.g., "cozy cottagecore picnic")');
        return;
      }

      showLoading();
      
      const btn = document.querySelector('.vibe-card .search-btn');
      const prevText = btn.innerText;
      btn.innerText = 'Searching...'; 
      btn.disabled = true;

      try {
        console.log('🔍 Searching for vibe:', vibeText);
        
        const response = await fetch(BACKEND_URL + '/search/vibe', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            vibe: vibeText, 
            max_results: 9
          })
        });
        
        if (!response.ok) {
          throw new Error(`Backend responded with status: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('✅ Received search results:', data);
        
        if (data && data.products && data.products.length > 0) {
          renderProducts(data.products);
          console.log(`📦 Rendered ${data.products.length} products`);
        } else {
          showError('No products found for this vibe. Try different keywords like "bohemian", "minimalist", or "vintage".');
        }
        
      } catch (error) {
        console.error('❌ Error during vibe search:', error);
        
        if (error.message.includes('Failed to fetch')) {
          showError('Cannot connect to the backend server. Please ensure the server is running on http://localhost:8000');
        } else if (error.message.includes('CORS')) {
          showError('CORS error: Backend server needs to allow requests from this origin.');
        } else {
          showError(`Search failed: ${error.message}`);
        }
      } finally {
        btn.innerText = prevText; 
        btn.disabled = false;
      }
    }

    // FIXED: Main search trigger
    function triggerSearch() {
      const query = document.getElementById('mainSearch').value.trim();
      if (!query) {
        showError('Please enter a search query');
        return;
      }
      
      document.getElementById('vibeInput').value = query;
      submitVibe();
      
      document.getElementById('vibe').scrollIntoView({ behavior: 'smooth' });
    }

    // FIXED: Load trends with better error handling
    async function loadTrends() {
      const chipsArea = document.getElementById('chipsArea');
      const featuredRow = document.getElementById('featuredRow');
      
      chipsArea.innerHTML = '<div class="loading-spinner" style="display:block;"></div>';
      featuredRow.innerHTML = '<div class="loading-spinner" style="display:block;"></div>';

      try {
        console.log('📈 Loading trending data...');
        
        const trendsResponse = await fetch(BACKEND_URL + '/trending');
        
        if (!trendsResponse.ok) {
          throw new Error(`Trends API responded with status: ${trendsResponse.status}`);
        }
        
        const trendsData = await trendsResponse.json();
        console.log('✅ Received trends data:', trendsData);
        
        if (trendsData && trendsData.trending_vibes) {
          renderChips(trendsData.trending_vibes.slice(0, 6));
        } else {
          chipsArea.innerHTML = '<div class="muted">No trending vibes available</div>';
        }
        
        if (trendsData && trendsData.featured_products) {
          renderFeatured(trendsData.featured_products.slice(0, 4));
        } else {
          featuredRow.innerHTML = '<div class="muted">No featured products available</div>';
        }
        
      } catch (error) {
        console.error('❌ Error loading trends:', error);
        chipsArea.innerHTML = '<div class="error-message">Failed to load trending vibes</div>';
        featuredRow.innerHTML = '<div class="error-message">Failed to load featured products</div>';
      }
    }

    // FIXED: Image search functionality
    async function searchByImage() {
      const fileInput = document.getElementById("imageUpload");
      if (!fileInput.files.length) {
        showError("Please upload an image!");
        return;
      }

      showLoading();
      
      // Update button state
      const uploadBtn = document.querySelector('.upload-btn');
      const originalText = uploadBtn.innerHTML;
      uploadBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Analyzing...';
      uploadBtn.style.pointerEvents = 'none';
      
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      formData.append("max_results", "9"); // Add max results parameter

      try {
        console.log('📷 Starting image search...');
        
        const response = await fetch(`${BACKEND_URL}/search/image`, {
          method: "POST",
          body: formData
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Image search failed: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        console.log('✅ Image search response:', data);
        
        if (data && data.products && data.products.length > 0) {
          renderProducts(data.products);
          console.log(`🎨 Found ${data.products.length} products matching your image style`);
          
          // Show analysis if available
          if (data.analysis && data.analysis.vibe_keywords) {
            console.log('🔍 Detected vibes:', data.analysis.vibe_keywords.join(', '));
          }
        } else {
          showError("No products found matching the uploaded image. Try a different image with clear fashion/style elements.");
        }
        
      } catch (error) {
        console.error('❌ Image search error:', error);
        
        // Provide specific error messages
        if (error.message.includes('503')) {
          showError('Image search requires Gemini API. Please check if GEMINI_API_KEY is configured.');
        } else if (error.message.includes('Failed to fetch')) {
          showError('Cannot connect to the backend server. Please ensure the server is running.');
        } else {
          showError(`Image search failed: ${error.message}`);
        }
      } finally {
        // Restore button state
        uploadBtn.innerHTML = originalText;
        uploadBtn.style.pointerEvents = 'auto';
      }
    }

    async function searchEventStyleInTrendRadar(styleKeywords) {
  try {
    const response = await fetch(BACKEND_URL + '/search/vibe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        vibe: styleKeywords.join(' '), 
        max_results: 6 
      })
    });
    
    const data = await response.json();
    
    // Ensure results container exists in trend radar
    const trendCard = document.querySelector('.trend-card');
    let resultsContainer = document.getElementById('trendResults');
    if (!resultsContainer) {
      resultsContainer = document.createElement('div');
      resultsContainer.id = 'trendResults';
      resultsContainer.innerHTML = '<div class="section-title" style="font-size:16px; margin-top:18px;">Event Products</div><div id="trendGrid" class="grid"></div>';
      trendCard.appendChild(resultsContainer);
    }
    
    // Update the title
    resultsContainer.querySelector('.section-title').textContent = 'Event Style Products';
    
    if (data.products && data.products.length > 0) {
      renderProducts(data.products, 'trendGrid');
    }
  } catch (error) {
    console.error('Event style search failed:', error);
  }
}

    // FIXED: Image upload preview and automatic search
    function previewUpload(e) {
      const file = e.target.files[0];
      if (!file) return;
      
      // Validate file type
      if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (JPG, PNG, etc.)');
        return;
      }
      
      // Check file size (optional - limit to 5MB)
      if (file.size > 5 * 1024 * 1024) {
        showError('Image too large. Please select an image smaller than 5MB.');
        return;
      }
      
      console.log('📷 Image uploaded:', file.name, 'Size:', (file.size / 1024 / 1024).toFixed(2) + 'MB');
      
      const reader = new FileReader();
      reader.onload = (ev) => {
        // Show preview with uploaded image
        const preview = {
          id: 999, 
          title: "📸 Your inspiration image", 
          price: "", 
          vibe_tags: "analyzing style...", 
          image_url: ev.target.result
        };
        renderProducts([preview]);
        
        // Show loading message
        setTimeout(() => {
          showSuccess('Image uploaded successfully! Analyzing style and finding matches...');
        }, 100);
        
        // Automatically trigger image search after a short delay
        setTimeout(() => {
          searchByImage();
        }, 500);
      };
      
      reader.onerror = () => {
        showError('Error reading the image file. Please try again.');
      };
      
      reader.readAsDataURL(file);
    }

    // FIXED: Enhanced page initialization
    document.addEventListener('DOMContentLoaded', async () => {
      console.log('🚀 Initializing Myntra Vibe Shopping...');
      
      // Test backend connectivity first
      try {
        const healthResponse = await fetch(BACKEND_URL + '/health');
        if (healthResponse.ok) {
          const healthData = await healthResponse.json();
          console.log('✅ Backend is healthy:', healthData);
        }
      } catch (error) {
        console.warn('⚠️ Backend health check failed:', error);
        showError('Backend server might not be running. Please start the server and refresh the page.');
      }
      
      // Load initial data
      await loadTrends();
      
      // Perform initial search
      document.getElementById('vibeInput').value = 'bohemian';
      await submitVibe();
      
      console.log('✅ App initialization complete');
    });

    // Add Enter key support for search inputs
    document.getElementById('mainSearch').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        triggerSearch();
      }
    });

    document.getElementById('vibeInput').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        submitVibe();
      }
    });