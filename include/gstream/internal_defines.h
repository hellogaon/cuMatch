#ifndef GSTREAM_INTERNAL_DEFINES_H
#define GSTREAM_INTERNAL_DEFINES_H
#include <stdint.h>

namespace gstream {

/**************************
 * Internal typedefs
 *************************/
using gaddr_t = uint64_t;
using laddr_t = uint32_t;
using grid_off = uint32_t;
using device_id_t = int32_t;
constexpr device_id_t InvalidDeviceID = -1;
constexpr device_id_t MultipleDevices = -2;

enum class degree_type: char {
    NIL = 0,
    InDegree = 1,
    OutDegree
};

enum class degree_type_code: char {
    NIL = 0,
    U8 = 8,
    U16 = 16,
    U24 = 24,
    U32 = 32,
    U64 = 64
};

inline unsigned char degree_type_code_to_stream_unit_size_byte(degree_type_code const& code) noexcept {
    static unsigned char const table[] = { 0, 1, 2, 4, 4, 8 };
    return table[static_cast<unsigned char>(code)/8];
}

class gstream_return_code;
char const* return_code_to_string(gstream_return_code const& rc) noexcept;

/* Return code */
class gstream_return_code {
public:
    enum {
        /* (Important) If you modify the gstream_return_code enum,
         * you also need to modify GSTREAM_RETCODE_SEQ in detail/runtime_supplementaries.cpp
         * in order for the enum to string functionality to work correctly.
         */
        Undefined = 0,
        Success,

        InvalidVirtualDeviceCount,
        InvalidRealDeviceID,
        InvalidHostStreamBufferSize,
        InvalidDeviceStreamBufferSize,
        InvalidNumDeviceStreams,
        InvalidGridFileStream,
        InsufficientHostStreamBufferSize,
        InsufficientDeviceStreamBufferSize,
        AlreadyPrefetched,

        NoCudaDevicesFound,
        DeviceCountError,
        SetDeviceError,
        BadallocDeviceMemory,
        BadallocPinnedMemory,
        BadallocPageableMemory,
        BadfreeDeviceMemory,
        BadfreePinnedMemory,
        BadfreePageableMemory,
        CacheControllerInitFailure,
        CommunicationModuleInitFailure,
        MsgListCreationFailure,
        EventListenerInitFailure,

        CudaError,
        LogicError,
        InvalidCall,
        InvalidArgument,
        WritePolicyMismatch,
        MemoryLeak,
        MessagePostFailure,
        DiskReadError,
        InvalidBufferSize,
        FileNotFound,
        FileOpenError,
        FileReadError,
        FileWriteError,
        Uninitialized,
        InitFailure,
        ChannelBroken,
        ReinitialzationError,
        DoubleFreeError,

        TXQueueEmpty,
        SchedulingError,

        BadallocBuddy,
        IOStall,

        OutdegreeLoadingError,
        IndegreeLoadingError,
        OutdegreeEmbedError,
        IndegreeEmbedError,

        NoIdleStream,

        UnhandledException,
        TransactionFault,
        IOHandingError,

        FatalError,
        NotImplemented
    };

    gstream_return_code() {
        value = 0;
    }

    gstream_return_code(gstream_return_code const& other) {
        value = other.value;
    }

    gstream_return_code(int const ivalue) {
        value = ivalue;
    }

    operator int() const noexcept {
        return value;
    }

    operator bool() const noexcept {
        return value == Success;
    }

    bool operator!() const noexcept {
        return value != Success;
    }

    gstream_return_code& operator=(int const new_value) noexcept {
        value = new_value;
        return *this;
    }

    bool operator==(gstream_return_code const& rhs) const noexcept {
        return value == rhs.value;
    }

    bool operator==(int const rhs) const noexcept {
        return value == rhs;
    }

    bool operator!=(gstream_return_code const& rhs) const noexcept {
        return value != rhs.value;
    }

    bool operator!=(int const rhs) const noexcept {
        return value != rhs;
    }

    char const* to_string() const noexcept {
        return return_code_to_string(*this);
    }

    int value;
};

constexpr uint32_t InternalDataSectionAlignment = 4;
constexpr uint32_t CudaMallocAlignment = 256;
constexpr uint32_t DegreeAlignment = 8;
constexpr uint32_t DiskIoSectorSize = 4096;
constexpr uint32_t MaxArity = 16;

struct kernel_launch_parameters {
    uint32_t num_threads_per_block;
    uint32_t num_blocks_per_grid;
    uint32_t shared_mem_size;
};

#define ADJ_FORMAT "part-%05d"
#define EL32_TEMPORARY_PATH_FORMAT "el32_tmp" // for LGF
#define EL32_SUBDIR_FORMAT "%d"
#define EL32_OPTIMAL_FILE_FORMAT "%d-%d.el32.optimal"
#define EL32_UNSORTED_FILE_FORMAT "%d-%d.el32.unsorted"
#define EL32_LABELED_FILE_FORMAT "%d-%d-%d.el32.labeled"

#pragma pack(push, 1)
struct manual_attr_option {
    bool use_in_degree : 1;
    bool use_out_degree : 1;
    bool use_manual_attr : 1;
    bool fixed_size : 1;
};
#pragma pack(pop)

enum class write_policy_type: unsigned char {
    Manual = 0,
    FixedColumnWise,
    FixedDeviceLocal,
    DynamicColumnWise,  // Not supported yet
    DynamicDeviceLocal, // Not supported yet
    PerKernel, // Not supported yet
    INFINEL,   // Not supported yet
};

inline bool required_batch_synchronization(write_policy_type const& mode) {
    switch (mode) {
    case write_policy_type::FixedColumnWise:
    case write_policy_type::FixedDeviceLocal:
    case write_policy_type::DynamicColumnWise:
    case write_policy_type::DynamicDeviceLocal:
        return true;
    default:
        return false;
    }
}

struct superstep_progress;
struct superstep_report;
class kernel_binder;
class gstream_query;

using FixedDegreeUnitType = uint32_t;
constexpr unsigned FixedDegreeUnitSize = sizeof(FixedDegreeUnitType); //TODO: Remove it, this value is temporary setting

namespace write_polices {

struct fixed_column_wise  {
    uint64_t columnar_output_size;
};
struct fixed_device_local {
    uint64_t output_unit_size;
};
struct per_kernel {
    int unused;
};

struct infinel {
    int unused;
};

} // namespace write_polices

struct write_policy {
    write_policy_type wp_type;
    union {
        write_polices::fixed_column_wise fixed_column_wise;
        write_polices::fixed_device_local fixed_device_local;
        write_polices::per_kernel per_kernel;
    };
};

namespace wb_configs {

struct fixed_column_wise {
    uint64_t buffer_size;
    grid_off col_id;
} ;

struct fixed_device_local {
    using key_type = uint32_t;
    uint64_t buffer_size;
    key_type access_key;
};

struct per_kernel {
    uint64_t buffer_size;
};

struct infinel {
    uint64_t buffer_size;
};

} // namespace wb_configs

struct device_buffer_descriptor {
    device_id_t rdev_id;
    device_id_t vdev_id;
    void* addr;
    //TODO: uint64_t bufsize;
};

struct write_buffer_configuration {
    write_policy_type wb_type;
    union {
        wb_configs::fixed_column_wise fixed_column_wise;
        wb_configs::fixed_device_local fixed_device_local;
        wb_configs::per_kernel per_kernel;
    };
};

struct write_buffer_descriptor {
    struct {
        bool wb_init : 1;
        bool wb_sync : 1;
    } flags;
    device_buffer_descriptor* devbuf_desc_arr;
    device_buffer_descriptor  _internal_devbuf_desc;
    write_buffer_configuration const* wb_cfg;
    device_id_t num_virtual_devcies;
    uint16_t num_cooperate_devices;
};

namespace wsync_configs {
namespace criteria_args {

struct column_wise {
    gstream_query* query;
};

struct device_local {
    uint64_t num_total_tx;
};

} // namespace criteria_args

struct sync_criteria {
    write_policy_type write_policy;
    union {
        criteria_args::column_wise column_wise;
        criteria_args::device_local device_local;
    };
};

} // namespace wsync_configs

using gridtx_id = int64_t;

namespace framework3 {

class grid_transaction;
struct framework3_instance;

} // !namespace framework3

enum class startup_mode: unsigned {
    _Undefined = 0,
    ColdStart,
    WarmStart,
};


struct memory_allocation_strategy {
    enum class alloc_mode: unsigned {
        _Undefined = 0,
        IntegrateBuddy, // IB
        SeparateBuddy,  // SB
        SeparateFixedGrid, // SFG
    };
    static alloc_mode string_to_alloc_mode(char const* mode_str);
    alloc_mode mode = alloc_mode::_Undefined;

    struct config_IntegrateBuddy {
        uint64_t buffer_size;
    };

    struct config_SeparateBuddy {
        uint64_t grid_buffer_size;
        uint64_t attr_buffer_size;
        uint32_t attr_alignemnt;
    };

    using config_SeparateFixedGrid = config_SeparateBuddy;

    union configuration {
        config_IntegrateBuddy IntegrateBuddy;
        config_SeparateBuddy SeparateBuddy;
        config_SeparateFixedGrid SeparateFixedGrid;
    } config;
};

} // !namespace gstream

#endif // GSTREAM_INTERNAL_DEFINES_H
