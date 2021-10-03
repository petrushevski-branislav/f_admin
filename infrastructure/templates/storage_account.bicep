param name string = 'fadminsa'

resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: name
  location: 'westeurope'
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowCrossTenantReplication: false
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: true
    allowSharedKeyAccess: true
    networkAcls: {
      bypass: 'AzureServices'
      virtualNetworkRules: []
      ipRules: []
      defaultAction: 'Allow'
    }
    supportsHttpsTrafficOnly: true
    encryption: {
      services: {
        file: {
          keyType: 'Account'
          enabled: true
        }
        blob: {
          keyType: 'Account'
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    accessTier: 'Cool'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2021-04-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    changeFeed: {
      enabled: false
    }
    restorePolicy: {
      enabled: false
    }
    containerDeleteRetentionPolicy: {
      enabled: true
      days: 30
    }
    cors: {
      corsRules: []
    }
    deleteRetentionPolicy: {
      enabled: true
      days: 30
    }
    isVersioningEnabled: false
  }
}

resource blobContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2021-04-01' = {
  parent: blobService
  name: 'files'
  properties: {
    defaultEncryptionScope: '$account-encryption-key'
    denyEncryptionScopeOverride: false
    publicAccess: 'Blob'
  }
}

// resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2021-04-01' = {
//   parent: storageAccount
//   name: 'default'
//   properties: {
//     protocolSettings: {
//       smb: {}
//     }
//     cors: {
//       corsRules: []
//     }
//     shareDeleteRetentionPolicy: {
//       enabled: false
//       days: 0
//     }
//   }
// }

// resource queueService 'Microsoft.Storage/storageAccounts/queueServices@2021-04-01' = {
//   parent: storageAccount
//   name: 'default'
//   properties: {
//     cors: {
//       corsRules: []
//     }
//   }
// }

// resource tableService 'Microsoft.Storage/storageAccounts/tableServices@2021-04-01' = {
//   parent: storageAccount
//   name: 'default'
//   properties: {
//     cors: {
//       corsRules: []
//     }
//   }
// }
