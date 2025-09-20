// Backend URL configuration
    const BACKEND_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
      ? 'http://localhost:8000' 
      : 'http://localhost:8000'; 

    console.log('🚀 Starting Myntra Vibe Shopping with Image Preview...');

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

    // ======================
    // IMAGE PREVIEW SYSTEM
    // ======================

    // Create the overlay HTML structure
    function createImageOverlay() {
        console.log('🖼️ Creating image overlay...');
        
        // Check if overlay already exists
        if (document.getElementById('imagePreviewOverlay')) {
            console.log('⚠️ Overlay already exists');
            return;
        }

        const overlayHTML = `
            <div id="imagePreviewOverlay" style="
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.8);
                z-index: 10000;
                display: none;
                justify-content: center;
                align-items: center;
                backdrop-filter: blur(3px);
            ">
                <div id="imagePreviewContainer" style="
                    position: relative;
                    max-width: 90vw;
                    max-height: 90vh;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                    overflow: hidden;
                    transform: scale(0.8);
                    opacity: 0;
                    transition: all 0.3s ease;
                ">
                    <!-- Close button -->
                    <button id="closePreview" style="
                        position: absolute;
                        top: 15px;
                        right: 15px;
                        width: 35px;
                        height: 35px;
                        border: none;
                        background: rgba(255, 255, 255, 0.9);
                        border-radius: 50%;
                        cursor: pointer;
                        z-index: 10001;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 18px;
                        font-weight: bold;
                        color: #666;
                        transition: all 0.2s;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
                    ">×</button>

                    <!-- Image container -->
                    <div style="
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        padding: 20px;
                        max-height: 90vh;
                        overflow: auto;
                    ">
                        <img id="previewImage" style="
                            max-width: 100%;
                            max-height: 60vh;
                            object-fit: contain;
                            border-radius: 8px;
                            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                        " />
                        
                        <!-- Product info -->
                        <div id="previewInfo" style="
                            margin-top: 15px;
                            text-align: center;
                            max-width: 400px;
                            padding: 15px;
                            background: #f9f9f9;
                            border-radius: 8px;
                            border: 1px solid #eee;
                        ">
                            <h3 id="previewTitle" style="
                                margin: 0 0 8px 0;
                                color: #333;
                                font-size: 18px;
                                font-weight: 600;
                            "></h3>
                            <div id="previewPrice" style="
                                font-size: 20px;
                                font-weight: bold;
                                color: #ff6b6b;
                                margin-bottom: 8px;
                            "></div>
                            <div id="previewTags" style="
                                font-size: 13px;
                                color: #666;
                                font-style: italic;
                                line-height: 1.4;
                            "></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add overlay to document
        document.body.insertAdjacentHTML('beforeend', overlayHTML);
        
        // Setup event listeners
        setupOverlayEventListeners();
        
        console.log('✅ Image preview overlay created successfully');
    }

    // Setup event listeners for the overlay
    function setupOverlayEventListeners() {
        const overlay = document.getElementById('imagePreviewOverlay');
        const closeBtn = document.getElementById('closePreview');
        const container = document.getElementById('imagePreviewContainer');
        
        if (!overlay) {
            console.error('❌ Overlay not found for event listeners');
            return;
        }
        
        console.log('🎯 Setting up overlay event listeners...');
        
        // Close on overlay background click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                console.log('📱 Clicked outside, closing preview');
                closeImagePreview();
            }
        });
        
        // Close on close button click
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                console.log('❌ Close button clicked');
                closeImagePreview();
            });
        }
        
        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && overlay.style.display === 'flex') {
                console.log('⌨️ Escape key pressed');
                closeImagePreview();
            }
        });
        
        // Prevent container clicks from closing overlay
        if (container) {
            container.addEventListener('click', (e) => {
                e.stopPropagation();
            });
        }
        
        console.log('✅ Event listeners set up successfully');
    }

    // Function to open image preview
    function openImagePreview(imageUrl, productInfo = {}) {
        console.log('🖼️ Opening image preview:', imageUrl, productInfo);
        
        // Ensure overlay exists
        createImageOverlay();
        
        const overlay = document.getElementById('imagePreviewOverlay');
        const container = document.getElementById('imagePreviewContainer');
        const previewImage = document.getElementById('previewImage');
        const previewTitle = document.getElementById('previewTitle');
        const previewPrice = document.getElementById('previewPrice');
        const previewTags = document.getElementById('previewTags');
        
        if (!overlay || !previewImage) {
            console.error('❌ Preview overlay elements not found');
            return;
        }
        
        // Set image source
        previewImage.src = imageUrl;
        previewImage.alt = productInfo.title || 'Product Image';
        
        // Set product info
        if (previewTitle) previewTitle.textContent = productInfo.title || 'Product';
        if (previewPrice) previewPrice.textContent = productInfo.price ? `₹${productInfo.price}` : '';
        
        // Handle tags
        let tagsText = '';
        if (productInfo.vibe_tags) {
            tagsText = Array.isArray(productInfo.vibe_tags) 
                ? productInfo.vibe_tags.join(', ') 
                : productInfo.vibe_tags;
        }
        if (previewTags) previewTags.textContent = tagsText;
        
        // Show overlay with animation
        overlay.style.display = 'flex';
        
        // Animate in
        setTimeout(() => {
            if (container) {
                container.style.transform = 'scale(1)';
                container.style.opacity = '1';
            }
        }, 10);
        
        // Prevent body scrolling when overlay is open
        document.body.style.overflow = 'hidden';
        
        // Handle image load errors
        previewImage.onerror = function() {
            console.warn('⚠️ Preview image failed to load');
            this.src = 'https://via.placeholder.com/400x400/f5f5f5/999999?text=Image+Not+Found';
        };
        
        previewImage.onload = function() {
            console.log('✅ Preview image loaded successfully');
        };
        
        console.log('✅ Image preview opened');
    }

    async function searchLocationEvents() {
  console.log('📍 Searching location events...');
  
  const locationInput = document.getElementById('locationInput');
  if (!locationInput) {
    showError('Location input not found');
    return;
  }

  const locationText = locationInput.value.trim();
  if (!locationText) {
    showError('Please enter a location to search events');
    return;
  }

  showLoading();

  try {
    const response = await fetch(BACKEND_URL + '/search/location-events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ location: locationText })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    console.log('📡 Location events data:', data);

    const products = data.products || data.results || [];
    if (products.length > 0) {
      renderProducts(products);
    } else {
      showError('No events found for this location');
    }
  } catch (error) {
    console.error('❌ Location search failed:', error);
    showError(`Search failed: ${error.message}`);
  }
}
window.searchLocationEvents = searchLocationEvents;


    // Function to close image preview
    function closeImagePreview() {
        const overlay = document.getElementById('imagePreviewOverlay');
        const container = document.getElementById('imagePreviewContainer');
        
        if (!overlay || overlay.style.display === 'none') {
            return;
        }
        
        console.log('🔒 Closing image preview...');
        
        // Animate out
        if (container) {
            container.style.transform = 'scale(0.8)';
            container.style.opacity = '0';
        }
        
        // Hide after animation completes
        setTimeout(() => {
            overlay.style.display = 'none';
            
            // Restore body scrolling
            document.body.style.overflow = 'auto';
            
            console.log('✅ Image preview closed');
        }, 300);
    }

    // ======================
    // PRODUCT RENDERING
    // ======================

    // Show error message
    function showError(message, containerId = 'resultsGrid') {
        console.log(`❌ Showing error in ${containerId}:`, message);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div class="error-message" style="padding: 20px; background: #fee; border: 1px solid #fcc; color: #c33; border-radius: 8px; margin: 16px 0;">⚠️ ${message}</div>`;
        }
    }

    // Show success message
    function showSuccess(message, containerId = 'resultsGrid') {
        console.log(`✅ Showing success in ${containerId}:`, message);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div class="success-message" style="padding: 20px; background: #efe; border: 1px solid #cfc; color: #363; border-radius: 8px; margin: 16px 0;">✅ ${message}</div>`;
        }
    }

    // Show loading spinner
    function showLoading(containerId = 'resultsGrid') {
        console.log(`⏳ Showing loading in ${containerId}`);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
              <div style="display: flex; justify-content: center; align-items: center; padding: 40px; font-size: 18px; color: #666;">
                <div style="border: 3px solid #f3f3f3; border-top: 3px solid #ff6b6b; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin-right: 12px;"></div>
                Loading...
              </div>
              <style>
                @keyframes spin {
                  0% { transform: rotate(0deg); }
                  100% { transform: rotate(360deg); }
                }
              </style>
            `;
        }
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
          renderFeatured(data.products, 'featuredRow');
          console.log(`🎭 Found ${data.products.length} products for events in ${location}`);
        } else {
          document.getElementById('featuredRow').innerHTML = 
            '<div style="padding: 20px; color: #666; text-align: center; font-size: 13px;">No suitable products found for events in this location.</div>';
        }
        
      } catch (error) {
        console.error('❌ Error during location event search:', error);
        
        // Hide event info on error
        document.getElementById('eventInfo').style.display = 'none';
        document.getElementById('eventChips').innerHTML = '';
        
        if (error.message.includes('Failed to fetch')) {
          document.getElementById('featuredRow').innerHTML = 
            '<div style="padding: 16px; color: #ef4444; text-align: center; font-size: 13px;">Cannot connect to server. Please check if backend is running.</div>';
        } else {
          document.getElementById('featuredRow').innerHTML = 
            '<div style="padding: 16px; color: #ef4444; text-align: center; font-size: 13px;">Event search failed. Please try again.</div>';
        }
      } finally {
        btn.innerHTML = prevText; 
        btn.disabled = false;
      }
    }

    // ======================
    // SEARCH FUNCTIONALITY
    // ======================

    // Enhanced vibe search
    async function submitVibe() {
        console.log('🔍 Submit vibe called');
        
        const vibeInput = document.getElementById('vibeInput');
        if (!vibeInput) {
            console.error('❌ Vibe input not found');
            return;
        }

        const vibeText = vibeInput.value.trim();
        if (!vibeText) {
            showError('Please enter a vibe to search');
            return;
        }

        showLoading();
        
        const btn = document.querySelector('.vibe-card .search-btn');
        let prevText = 'Find Vibes';
        if (btn) {
            prevText = btn.innerText;
            btn.innerText = 'Searching...';
            btn.disabled = true;
        }

        try {
            console.log('📡 Searching for:', vibeText);
            
            const response = await fetch(BACKEND_URL + '/search/vibe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ vibe: vibeText, max_results: 9 })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📡 Response data:', data);
            
            const products = data.products || data.results || [];
            console.log('📦 Extracted products:', products);
            
            if (products.length > 0) {
                showSuccess(`Found ${products.length} products matching "${vibeText}"`);
                setTimeout(() => renderProducts(products), 100);
            } else {
                showError('No products found. Try different keywords.');
            }
            
        } catch (error) {
            console.error('❌ Search failed:', error);
            showError(`Search failed: ${error.message}`);
        } finally {
            if (btn) {
                btn.innerText = prevText;
                btn.disabled = false;
            }
        }
    }

    // Renders trend hashtags
// Renders trend hashtags
function renderChips(trends) {
    const chipsArea = document.getElementById('chipsArea');
    if (!chipsArea) return;
    chipsArea.innerHTML = '';

    // Create the container with the .chips class
    const chipsContainer = document.createElement('div');
    chipsContainer.className = 'chips';

    trends.forEach(trend => {
        const chip = document.createElement('div');
        chip.className = 'chip';
        chip.textContent = `#${trend}`;

        chip.onclick = () => {
            document.getElementById('vibeInput').value = trend;
            submitVibe();
        };
        chipsContainer.appendChild(chip);
    });

    // Append the entire container to the chipsArea
    chipsArea.appendChild(chipsContainer);
}

// Renders featured products
function renderFeatured(products) {
    const featuredRow = document.getElementById('featuredRow');
    if (!featuredRow) return;
    featuredRow.innerHTML = '';
    products.forEach(product => {
        const item = document.createElement('div');
        item.className = 'featured-item';
        item.innerHTML = `
            <img src="${product.image_url}" alt="${escapeHtml(product.title)}" />
            <div class="ft-meta">
                <div class="small">${escapeHtml(product.title)}</div>
                <div class="price" style="font-size:14px;">₹${product.price}</div>
            </div>
        `;
        item.onclick = () => {
            openImagePreview(product.image_url, product);
        };
        featuredRow.appendChild(item);
    });
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

    // ======================
    // INITIALIZATION
    // ======================

    // Make functions globally available
    window.submitVibe = submitVibe;
    window.openImagePreview = openImagePreview;
    window.closeImagePreview = closeImagePreview;

    // Test function for debugging
    window.testPreview = () => {
        console.log('🧪 Testing image preview...');
        openImagePreview(
            'https://via.placeholder.com/400x400/ff6b6b/ffffff?text=TEST+IMAGE',
            {
                title: 'Test Product',
                price: '1299',
                vibe_tags: 'test, preview, demo'
            }
        );
    };

    console.log('✅ Main.js loaded with image preview functionality!');
    console.log('🧪 Run testPreview() in console to test the preview system');

    // Auto-test after page loads
    setTimeout(() => {
        const vibeInput = document.getElementById('vibeInput');
        if (vibeInput && !vibeInput.value) {
            vibeInput.value = 'bohemian';
            submitVibe();
        }
    }, 1000);

    // Add this at the end of your main.js file
document.addEventListener('DOMContentLoaded', () => {
  loadTrends();
});