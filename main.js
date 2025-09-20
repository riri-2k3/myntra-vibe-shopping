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

    // ENHANCED: Product rendering with image preview
    // ENHANCED: Product rendering with image preview
function renderProducts(products, containerId = 'resultsGrid') {
    console.log('🎯 === RENDER PRODUCTS WITH PREVIEW ===');
    console.log('📦 Products:', products);
    console.log('📦 Container ID:', containerId);

    const container = document.getElementById(containerId);
    console.log('📦 Container found:', !!container);

    if (!container) {
        console.error(`❌ Container '${containerId}' not found!`);
        return;
    }

    // Clear container before rendering new products
    container.innerHTML = '';

    if (!products || products.length === 0) {
        container.innerHTML = `
              <div style="text-align: center; padding: 40px; color: #666;">
                <div style="font-size: 48px; margin-bottom: 16px;">🛍️</div>
                <div style="font-size: 18px;">No products found</div>
                <div style="font-size: 14px; color: #999; margin-top: 8px;">Try different keywords</div>
              </div>
            `;
        return;
    }

    console.log(`✅ Rendering ${products.length} products...`);

    products.forEach((product, index) => {
        console.log(`🎨 Rendering product ${index + 1}:`, product);

        // Extract product data
        const title = product.title || product.name || `Product ${index + 1}`;
        const price = product.price || 'N/A';
        const imageUrl = product.image_url || product.image || 'https://via.placeholder.com/300x300?text=No+Image';

        let vibeTagsText = '';
        if (product.vibe_tags) {
            vibeTagsText = Array.isArray(product.vibe_tags)
                ? product.vibe_tags.join(', ')
                : product.vibe_tags;
        }

        // Create product card with the correct class name "product"
        const card = document.createElement('div');
        card.className = 'product';

        // Build card HTML using the correct class names from style.css
        card.innerHTML = `
              <div class="imgwrap">
                <img
                  src="${imageUrl}"
                  alt="${escapeHtml(title)}"
                  onerror="this.src='https://via.placeholder.com/300x250/f5f5f5/999999?text=No+Image'"
                />
              </div>
              <div class="meta">
                <h4>${escapeHtml(title)}</h4>
                <div class="price">₹${price}</div>
                ${vibeTagsText ? `<div class="tags">${escapeHtml(vibeTagsText)}</div>` : ''}
              </div>
            `;

        // Add the click handler for image preview
        card.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            console.log('🖼️ Product card clicked, opening preview for:', title);
            openImagePreview(imageUrl, {
                title: title,
                price: price,
                vibe_tags: vibeTagsText
            });
        });

        // Append the product card directly to the existing container in the HTML
        container.appendChild(card);
        console.log(`✅ Product ${index + 1} added with click handler`);
    });

    console.log(`🎉 Successfully rendered ${products.length} products with image preview!`);
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